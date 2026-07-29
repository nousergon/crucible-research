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

log() { echo "[thinktank-bootstrap] $*"; }
fail() { echo "[thinktank-bootstrap] FATAL: $*" >&2; exit 1; }

# Self-terminate on EVERY exit path, preserving the real exit code for the SSM
# command status. `shutdown -h now` + InstanceInitiatedShutdownBehavior=terminate
# is the fleet's standard teardown.
on_exit() {
    local rc=$?
    log "run finished rc=${rc} — terminating box"
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
