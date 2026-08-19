#!/usr/bin/env bash
#
# thinktank_spot_bootstrap.sh — on-box entry for the daily Think Tank spot run
# (config-I5208 / nous-ergon-ops-I162, ARCHITECTURE §47).
#
# Invoked by the alpha-engine-thinktank-spot-dispatcher Lambda's SSM prelude,
# which has already installed git/python3.12 and cloned THIS repo. Everything
# version-controlled lives here rather than in the Lambda's command string, so
# a change to the run shape is a PR in this repo, not a Lambda redeploy.
#
# Contract:
#   * SELF-TERMINATES unconditionally. The trap fires on success, failure, and
#     SIGTERM (spot reclaim) alike — an orphaned box burns money silently and
#     the spot-orphan-reaper is a 6.5h AGE CAP, not a health check, so it is
#     not a substitute for shutting ourselves down.
#   * FAILS LOUD. No `|| true` on anything load-bearing; the exit code
#     propagates to the SSM command status, which is the dispatcher's only
#     honest failure signal.
#   * PROMPTS ARE MANDATORY. A fresh public clone has no prompts (gitignored
#     IP). prompt_loader hard-fails rather than falling back to samples (the
#     2026-04-11 silent-sample-fallback incident); this script checks for them
#     BEFORE spending any LLM budget so a misconfigured box dies in seconds
#     rather than after a partial run.
#
# The run budget is passed in by the dispatcher and MUST stay below its SSM
# execution_timeout — see thinktank_box_runner.py for why the deadline is
# load-bearing rather than advisory.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
REPO_DIR="${REPO_DIR:-/home/ec2-user/crucible-research}"
CONFIG_DIR="${CONFIG_DIR:-/home/ec2-user/alpha-engine-config}"
CONFIG_REPO="${CONFIG_REPO:-nousergon/alpha-engine-config}"
CONFIG_PAT_SSM="${CONFIG_PAT_SSM:-/alpha-engine/saturday_sf_watch/github_pat}"
ALPHA_ENGINE_EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
THINKTANK_RUN_BUDGET_SECONDS="${THINKTANK_RUN_BUDGET_SECONDS:-12600}"

# ── Terminal-outcome reporting (alpha-engine-config-I5752) ──────────────────
# The box self-terminates on every exit path, so anything not shipped off it
# before `shutdown` is gone. Measured 2026-07-30, the coverage this closes:
#
#   stage              | before                        | after
#   -------------------|-------------------------------|---------------------
#   dispatcher prelude | log -> S3, no alert            | unchanged (its own trap)
#   THIS script        | NOTHING — no log, no alert     | log -> S3 + SNS on failure
#   the Python run     | flow-doctor Telegram + email   | unchanged
#   overran the cap    | reaped, no incomplete-reap row | marker + WatchKind
#
# The middle row is the gap. `fail()` here is what refuses to start when the
# private prompt config is absent -- deliberately, before any LLM spend -- and
# that refusal was silent: stderr died with the box and the SSM command status
# is watched by nothing (alpha-engine-config-I5752).
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts}"
RUN_TOKEN="${THINKTANK_SPOT_RUN_TOKEN:-unknown}"
TRADING_DAY="$(date -u +%Y-%m-%d)"
BOOTSTRAP_LOG="${BOOTSTRAP_LOG:-/var/log/thinktank-spot-bootstrap-${RUN_TOKEN}.log}"
S3_LOG="s3://alpha-engine-research/_ssm_logs/thinktank-spot/${TRADING_DAY}/bootstrap-$(hostname)-${RUN_TOKEN}.log"

# SUCCESS PATH ONLY -- principles.md §2.7. This marker is what
# spot-orphan-reaper's Think-Tank WatchKind reads to decide whether a box it
# reaped at the fleet age cap had actually finished. A marker a failed run
# wrote would make the failure read as a completed run to the one detector
# that can see a hung box, which is the exact inversion §2.7 forbids.
COMPLETION_KEY="s3://alpha-engine-research/thinktank/_control/completed/${TRADING_DAY}-${RUN_TOKEN}.json"

log() { echo "[thinktank-bootstrap] $*"; }
fail() { echo "[thinktank-bootstrap] FATAL: $*" >&2; exit 1; }

# Self-terminate on EVERY exit path, preserving the real exit code for the SSM
# command status. `shutdown -h now` + InstanceInitiatedShutdownBehavior=terminate
# is the fleet's standard teardown.
#
# Every step here is best-effort and `|| true`: the run's outcome is already
# decided by `rc`, and a failure to REPORT it must never change it, nor block
# the shutdown that stops the box billing. Each swallow is visible in the log
# this same function ships (no-silent-fails carve-out: the failure mode is
# "the report did not land", the primary deliverable is the run itself, and
# the recording surface is $S3_LOG plus the SSM command status).
on_exit() {
    local rc=$?
    log "run finished rc=${rc} — reporting outcome, then terminating box"

    # Ship the log on EVERY exit, success included: a successful run's log is
    # the baseline the next failure gets diffed against.
    aws s3 cp "$BOOTSTRAP_LOG" "$S3_LOG" --region "$REGION" --quiet 2>/dev/null || true

    if [ "$rc" -eq 0 ]; then
        printf '{"trading_day":"%s","run_token":"%s","instance":"%s","completed_at":"%s"}\n' \
            "$TRADING_DAY" "$RUN_TOKEN" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            | aws s3 cp - "$COMPLETION_KEY" --region "$REGION" --quiet 2>/dev/null || true
        log "completion marker written: ${COMPLETION_KEY}"
    else
        log "NO completion marker — this run failed (rc=${rc})"
        aws sns publish --topic-arn "$SNS_TOPIC_ARN" --region "$REGION" \
            --subject "Think Tank box FAILED (rc=${rc})" \
            --message "The daily Think Tank box exited ${rc} before its run completed.
trading_day=${TRADING_DAY} run_token=${RUN_TOKEN} instance=$(hostname)
Log: ${S3_LOG}
Last 40 lines:
$(tail -n 40 "$BOOTSTRAP_LOG" 2>/dev/null || echo '(log unavailable)')

This alert covers the window flow-doctor cannot: a failure BEFORE the Python
run configures it (private config absent, venv, deps), where stderr dies with
the box. A failure inside the run reports itself through flow-doctor instead.
See alpha-engine-config-I5752." >/dev/null 2>&1 || true
    fi

    shutdown -h now >/dev/null 2>&1 || true
    exit "$rc"
}
trap on_exit EXIT

cd "$REPO_DIR" || fail "repo not found at $REPO_DIR (dispatcher prelude clone failed?)"

# ── Private config (prompts) ────────────────────────────────────────────────
# Package-first with legacy top-level fallback — mirrors deploy.sh's resolution
# and spot_research_weekly.sh's staging step. prompt_loader/config.py search
# path #1 is the HOME-sibling checkout, which is exactly where we clone it.
log "cloning private config for prompts (experiment=${ALPHA_ENGINE_EXPERIMENT_ID})"
PAT=$(aws ssm get-parameter --name "$CONFIG_PAT_SSM" --with-decryption \
    --query Parameter.Value --output text --region "$REGION") \
    || fail "could not read config PAT from ${CONFIG_PAT_SSM}"
[ -n "$PAT" ] || fail "config PAT empty"
git config --global --add safe.directory '*' || true
rm -rf "$CONFIG_DIR"
git clone --depth 1 "https://x-access-token:${PAT}@github.com/${CONFIG_REPO}.git" \
    "$CONFIG_DIR" >/dev/null 2>&1 || fail "config clone failed"
unset PAT

PROMPT_DIR=""
for candidate in \
    "$CONFIG_DIR/experiments/${ALPHA_ENGINE_EXPERIMENT_ID}/research/prompts" \
    "$CONFIG_DIR/research/prompts"; do
    if ls "$candidate"/thinktank_*.txt >/dev/null 2>&1; then PROMPT_DIR="$candidate"; break; fi
done
[ -n "$PROMPT_DIR" ] || fail "think-tank prompts not found under $CONFIG_DIR (deploy.sh parity check) — refusing to run against sample prompts"
log "prompts resolved: $PROMPT_DIR"

# ── DLP scan prerequisite (krepis>=0.59.15, krepis-PR95, nous-ergon-ops-I738) ─
# LLMClient shells out to gitleaks on every call and fails CLOSED when the
# binary or config dir is absent (session_dlp.py). requirements.txt pins
# krepis>=0.59.8 with no ceiling, so a fresh install here resolves the DLP-hook
# version same as everywhere else. The Lambda image (Dockerfile) and the
# groom/alert_drain/sf_watch spot bootstraps in alpha-engine-config already
# carry this step; this box never did, so every DLP-gated LLM call it makes
# raises "gitleaks binary not found on PATH" and the run dies with rc=1
# (alpha-engine-config-I<TBD>). Mirrors the Dockerfile's install exactly:
# /opt/llm-routing is session_dlp.py's first standard-path fallback, so no
# KREPIS_GITLEAKS_DIR override is needed.
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
log "building venv"
python3.12 -m venv .venv || fail "venv creation failed"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip || fail "pip upgrade failed"
pip install --quiet -r requirements.txt || fail "requirements install failed"

# ── Run ─────────────────────────────────────────────────────────────────────
export AWS_DEFAULT_REGION="$REGION"
export ALPHA_ENGINE_EXPERIMENT_ID
export THINKTANK_RUN_BUDGET_SECONDS
log "starting daily run (budget=${THINKTANK_RUN_BUDGET_SECONDS}s)"
python infrastructure/thinktank_box_runner.py
