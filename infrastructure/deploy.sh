#!/usr/bin/env bash
# deploy.sh — Build and deploy Lambda functions to AWS.
#
# Main function uses container image (10 GB limit) because dependencies
# exceed the 250 MB zip size limit (numpy + pandas + curl_cffi + yfinance).
# Alerts function uses zip (lightweight, no heavy deps).
#
# Prerequisites:
#   1. AWS CLI configured with appropriate credentials
#   2. IAM role created (alpha-engine-research-role)
#   3. S3 bucket created (alpha-engine-research)
#   4. ECR repository created: alpha-engine-research-runner
#   5. Docker installed and running
#
# Usage: ./infrastructure/deploy.sh [main|alerts|both]

set -euo pipefail

FUNCTION_MAIN="alpha-engine-research-runner"
FUNCTION_ALERTS="alpha-engine-research-alerts"
FUNCTION_EVAL_JUDGE="alpha-engine-research-eval-judge"
# Batch-API chain Lambdas (ROADMAP §1642 closure 2026-05-07).
FUNCTION_EVAL_JUDGE_SUBMIT="alpha-engine-research-eval-judge-submit"
FUNCTION_EVAL_JUDGE_POLL="alpha-engine-research-eval-judge-poll"
FUNCTION_EVAL_JUDGE_PROCESS="alpha-engine-research-eval-judge-process"
FUNCTION_EVAL_ROLLING_MEAN="alpha-engine-research-eval-rolling-mean"
FUNCTION_RATIONALE_CLUSTERING="alpha-engine-research-rationale-clustering"
FUNCTION_AGGREGATE_COSTS="alpha-engine-research-aggregate-costs"
FUNCTION_SCANNER="alpha-engine-research-scanner"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="alpha-engine-research"
BUILD_DIR="lambda/package"

# ECR repository for container image deployment
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || echo "ACCOUNT_ID")
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/alpha-engine-research-role}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_MAIN}"

TARGET="${1:-both}"


# ── Lambda existence check (fail-loud on non-NotFound errors) ────────────────
#
# Antipattern this replaces: ``if aws lambda get-function ... &>/dev/null``
# combined stdout+stderr redirect, so AccessDenied / 504 / throttle errors
# were silently swallowed and the script fell through to the create-function
# branch — surfacing as a confusing "Function already exist" downstream
# (alpha-engine-data#149 incident triage 2026-05-04 + eval-judge deploy
# transient AWS 504 on 2026-05-08). Closes ROADMAP P3 line ~133.
#
# Returns 0 if the function exists (caller proceeds to update path).
# Returns 1 if the function doesn't exist (caller proceeds to create path).
# Exits the script (non-zero) on any other error — AccessDenied / 504 /
# throttle / network — so the operator sees the real cause instead of the
# misleading downstream error. AWS 504 is intermittent; operator retries.

_lambda_function_exists() {
  local fn_name="$1"
  local err
  if err=$(aws lambda get-function \
        --function-name "$fn_name" \
        --region "$REGION" 2>&1 >/dev/null); then
    return 0
  fi
  if echo "$err" | grep -q -E "ResourceNotFoundException|Function not found"; then
    return 1
  fi
  echo "ERROR: aws lambda get-function failed for '$fn_name' with non-NotFound error:" >&2
  echo "$err" >&2
  echo "Hint: AccessDenied → check IAM policy on the calling principal." >&2
  echo "Hint: 504/throttle → transient AWS issue; retry the deploy." >&2
  exit 1
}

# ── Throttle-aware Lambda invoke (bounded, jittered retry) ───────────────────
#
# Reserved-concurrency=1 singleton guards (the research runner has ONE slot
# fleet-wide, so two research graphs can't run at once and double LLM spend /
# S3 writes) can have their lone slot legitimately occupied when the canary
# fires: an overlapping deploy's canary Lambda (cancelling a GitHub Actions run
# does NOT stop the Lambda execution it already dispatched — bit
# crucible-research CI 2026-07-01 when #347 and #348 merged back-to-back), or
# an in-flight scheduled invocation. AWS then returns TooManyRequestsException /
# ReservedFunctionConcurrentInvocationLimitExceeded, and the AWS CLI's own retry
# (max 2, seconds-scale) can't outwait an in-flight execution.
#
# This retries ONLY on that throttle/concurrency signal, with exponential
# backoff + jitter, bounded to ~3 min. A NON-throttle invoke error (bad
# payload, missing function, AccessDenied) is NOT retried — it returns
# immediately with the real stderr. Exhausting retries also returns non-zero
# (fail loud, per the no-silent-fails rule); the caller decides what a
# never-completed invoke means.
#
# Args: <output-file> <aws lambda invoke flags...>   (the output positional is
#       appended internally — pass only the flags in "$@").
# Returns: 0 once the invoke API call succeeds (payload in <output-file>);
#          non-zero on a non-throttle error or exhausted retries.
_invoke_lambda_with_throttle_retry() {
  local out_file="$1"; shift
  local max_attempts=6 attempt=1 rc base sleep_s err_file
  err_file=$(mktemp)
  while :; do
    rc=0
    aws lambda invoke "$@" "$out_file" >/dev/null 2>"$err_file" || rc=$?
    if [ "$rc" -eq 0 ]; then
      rm -f "$err_file"
      return 0
    fi
    if [ "$attempt" -lt "$max_attempts" ] && \
       grep -qE 'TooManyRequestsException|ReservedFunctionConcurrentInvocationLimitExceeded' "$err_file"; then
      base=$(( 2 ** (attempt - 1) * 5 ))   # 5, 10, 20, 40, 80s
      sleep_s=$(( base + RANDOM % 5 ))     # + 0-4s jitter
      echo "  Canary invoke throttled — reserved-concurrency slot busy (attempt ${attempt}/${max_attempts}); retrying in ${sleep_s}s..." >&2
      sleep "$sleep_s"
      attempt=$(( attempt + 1 ))
      continue
    fi
    echo "  ERROR: canary invoke failed (exit ${rc}) after ${attempt} attempt(s):" >&2
    cat "$err_file" >&2
    rm -f "$err_file"
    return "$rc"
  done
}

# ── Main function: container image deployment ────────────────────────────────

build_and_deploy_main() {
  echo "=== Building container image for $FUNCTION_MAIN ==="

  # alpha-engine-lib is installed inside the Dockerfile via pip from
  # public git+https (lib was flipped public 2026-05-03). No vendor
  # staging needed.
  rm -rf flow-doctor-pkg  # legacy path — remove any stale artifact from prior builds

  # Stage proprietary configs from the private alpha-engine-config repo
  # into the build context. Prompts, scoring.yaml, and universe.yaml are
  # gitignored in this repo (see .gitignore) so a fresh GitHub Actions
  # checkout has none of them — the image would ship broken (or worse,
  # silently fall back to the committed *.sample.yaml files and run on
  # trivial placeholder data, which is exactly what happened on the
  # 2026-04-11 research Lambda run).
  #
  # Local dev workflow is preserved: if the real files already exist in
  # config/ on the laptop, we use them as-is.
  CONFIG_REPO_DIR="${CONFIG_REPO_DIR:-$(dirname "$(pwd)")/alpha-engine-config}"
  PROMPTS_STAGED_FROM_CONFIG_REPO=0
  YAMLS_STAGED_FROM_CONFIG_REPO=()

  # -- prompts -------------------------------------------------------------
  if [ -d "config/prompts" ] && ls config/prompts/*.txt &>/dev/null; then
    echo "Using existing config/prompts/ (local dev workflow)"
  else
    if [ -d "$CONFIG_REPO_DIR/research/prompts" ]; then
      echo "Staging research prompts from $CONFIG_REPO_DIR/research/prompts/..."
      mkdir -p config/prompts
      cp "$CONFIG_REPO_DIR/research/prompts/"*.txt config/prompts/
      PROMPTS_STAGED_FROM_CONFIG_REPO=1
    else
      echo "ERROR: research prompts not found — tried:"
      echo "  config/prompts/ (local dev)"
      echo "  $CONFIG_REPO_DIR/research/prompts/ (config repo sibling)"
      echo "Hint: clone nousergon/alpha-engine-config as a sibling directory,"
      echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
      exit 1
    fi
  fi

  # -- scoring.yaml + universe.yaml ---------------------------------------
  for yaml in scoring.yaml universe.yaml; do
    if [ -f "config/$yaml" ]; then
      echo "Using existing config/$yaml (local dev workflow)"
    else
      src="$CONFIG_REPO_DIR/research/$yaml"
      if [ -f "$src" ]; then
        echo "Staging config/$yaml from $src..."
        cp "$src" "config/$yaml"
        YAMLS_STAGED_FROM_CONFIG_REPO+=("$yaml")
      else
        echo "ERROR: config/$yaml not found — tried:"
        echo "  config/$yaml (local dev)"
        echo "  $src (config repo sibling)"
        echo "Hint: clone nousergon/alpha-engine-config as a sibling directory,"
        echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
        exit 1
      fi
    fi
  done

  # -- model_pricing.yaml (cost telemetry) --------------------------------
  # Lives under cost/ in alpha-engine-config and gets flattened to
  # config/model_pricing.yaml in the Lambda image to match _find_config()'s
  # subdir-flattened search step.
  if [ -f "config/model_pricing.yaml" ]; then
    echo "Using existing config/model_pricing.yaml (local dev workflow)"
  else
    src="$CONFIG_REPO_DIR/cost/model_pricing.yaml"
    if [ -f "$src" ]; then
      echo "Staging config/model_pricing.yaml from $src..."
      cp "$src" "config/model_pricing.yaml"
      YAMLS_STAGED_FROM_CONFIG_REPO+=("model_pricing.yaml")
    else
      echo "ERROR: config/model_pricing.yaml not found — tried:"
      echo "  config/model_pricing.yaml (local dev)"
      echo "  $src (config repo sibling)"
      echo "Hint: clone nousergon/alpha-engine-config as a sibling directory,"
      echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
      exit 1
    fi
  fi

  # Stamp the image with the source commit SHA so the decision-capture
  # provenance stamp (DecisionArtifact.code_sha, L4567 sub-item 1b / #781)
  # records the exact deployed code. CI passes $GITHUB_SHA; a manual deploy
  # falls back to `git rev-parse HEAD`. Empty (not "unknown") when neither
  # resolves, so graph/research_graph.py's `os.environ.get(...) or None` read
  # records None rather than a misleading literal. Mirrors the predictor wire-in.
  GIT_SHA="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo '')}"
  echo "  Stamping image with GIT_SHA=${GIT_SHA:-<unset>}"

  # Build Docker image
  echo "Building Docker image..."
  docker build --platform linux/amd64 --provenance=false \
    --build-arg "GIT_SHA=${GIT_SHA}" \
    -t "$FUNCTION_MAIN:latest" .

  # Only remove staged files — never touch a local dev checkout that
  # already had real files present.
  if [ "$PROMPTS_STAGED_FROM_CONFIG_REPO" = "1" ]; then
    rm -rf config/prompts
  fi
  # Guard the array expansion — under `set -u`, expanding an empty array
  # with `[@]` raises "unbound variable" (Bash <4.4). The `[@]+...` pattern
  # only emits the elements when the array exists and is non-empty.
  for yaml in "${YAMLS_STAGED_FROM_CONFIG_REPO[@]+"${YAMLS_STAGED_FROM_CONFIG_REPO[@]}"}"; do
    rm -f "config/$yaml"
  done

  # Authenticate with ECR
  echo "Authenticating with ECR..."
  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

  # Ensure ECR repository exists
  aws ecr describe-repositories --repository-names "$FUNCTION_MAIN" --region "$REGION" &>/dev/null || \
    aws ecr create-repository --repository-name "$FUNCTION_MAIN" --region "$REGION" > /dev/null

  # Tag and push
  echo "Pushing image to ECR..."
  docker tag "$FUNCTION_MAIN:latest" "$ECR_REPO:latest"
  docker push "$ECR_REPO:latest"
  IMAGE_URI="$ECR_REPO:latest"

  # Update or create Lambda function
  echo "Deploying $FUNCTION_MAIN..."

  if _lambda_function_exists "$FUNCTION_MAIN"; then
    # Check if existing function is zip-based (can't switch to image in-place)
    EXISTING_PKG=$(aws lambda get-function-configuration \
      --function-name "$FUNCTION_MAIN" --region "$REGION" \
      --query "PackageType" --output text 2>/dev/null || echo "Zip")

    if [ "$EXISTING_PKG" = "Image" ]; then
      # Already container-based — update the image and env vars
      aws lambda update-function-code \
        --function-name "$FUNCTION_MAIN" \
        --image-uri "$IMAGE_URI" \
        --region "$REGION" > /dev/null
    else
      # Zip → Image migration: delete and recreate
      echo "  Migrating from zip to container image..."
      aws lambda delete-function --function-name "$FUNCTION_MAIN" --region "$REGION"
      sleep 2

      aws lambda create-function \
        --function-name "$FUNCTION_MAIN" \
        --package-type Image \
        --code "ImageUri=$IMAGE_URI" \
        --role "$ROLE_ARN" \
        --timeout 900 \
        --memory-size 1024 \
        --region "$REGION" > /dev/null

      echo "  NOTE: EventBridge triggers were removed with the old function."
      echo "  Re-run setup-eventbridge.sh to restore schedules."
    fi
  else
    # Fresh create
    aws lambda create-function \
      --function-name "$FUNCTION_MAIN" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --role "$ROLE_ARN" \
      --timeout 900 \
      --memory-size 1024 \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_MAIN deployed (container image)."

  # Publish version and update 'live' alias
  echo "  Publishing Lambda version..."
  aws lambda wait function-updated --function-name "$FUNCTION_MAIN" --region "$REGION" 2>/dev/null || sleep 5
  VERSION=$(aws lambda publish-version \
    --function-name "$FUNCTION_MAIN" \
    --query "Version" --output text \
    --region "$REGION")
  echo "  Published version: $VERSION"
  aws lambda update-alias \
    --function-name "$FUNCTION_MAIN" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "$FUNCTION_MAIN" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION"
  echo "  Alias 'live' → version $VERSION"

  # Canary invocation
  #
  # Use ``dry_run_llm: true`` — the flag the handler actually recognizes
  # (lambda/handler.py:191). Earlier versions sent ``{"dry_run": true}``,
  # which the handler silently ignored, leaving the canary running in
  # full production mode (real LLM calls, real S3 writes, real email).
  # That misfired on 2026-05-04 when a config-changed deploy landed
  # inside the 5:40-5:55 PT weekday gate window in
  # ``_is_scheduled_run_time()`` and produced a real ``signals.json`` +
  # research email outside the intended Saturday cadence. The
  # ``dry_run_llm`` path installs full stubs (no LLM, no S3, no email)
  # before the graph runs, so a future deploy landing in the gate
  # window stays a no-op.
  echo "  Running canary (dry_run_llm=true)..."
  CANARY_OUT=$(mktemp)
  if ! _invoke_lambda_with_throttle_retry "$CANARY_OUT" \
      --function-name "${FUNCTION_MAIN}:live" \
      --payload '{"dry_run_llm": true}' \
      --cli-binary-format raw-in-base64-out \
      --region "$REGION"; then
    # The invoke API never returned a payload — either a non-throttle error, or
    # the reserved-concurrency slot stayed busy past the bounded retry window.
    # The deploy itself SUCCEEDED (the live alias already moved to $VERSION); a
    # never-run smoke test is NOT a canary failure, so do NOT roll back —
    # reverting a healthy deploy because we couldn't get a test slot would be
    # the wrong action. Surface loud (fail the job) + alert so an operator
    # confirms the live version by hand. Distinct dedup-key from the
    # bad-STATUS rollback path below.
    rm -f "$CANARY_OUT"
    echo "  ERROR: canary could not be invoked (slot contention or invoke error) — deploy left LIVE on v${VERSION}, NOT rolled back."
    python3 -m krepis.alerts publish \
      --severity error \
      --source "alpha-engine-research/infrastructure/deploy.sh" \
      --dedup-key "canary-uninvokable-${FUNCTION_MAIN}-v${VERSION}" \
      --message "Canary could NOT be invoked for ${FUNCTION_MAIN} v${VERSION} (throttle/concurrency or invoke error, retries exhausted). Live alias LEFT on v${VERSION} — deploy succeeded, NOT rolled back. Verify the live version manually." \
      || true
    exit 1
  fi

  # Handler returns {"status": "OK|SKIPPED|ERROR"} or {"statusCode": 500} on env var failure.
  # Accept OK or SKIPPED (wrong_time / already_run / market_holiday are expected).
  CANARY_STATUS=$(python3 -c "
import json, sys
d = json.load(open('$CANARY_OUT'))
s = d.get('status', '')
if s in ('OK', 'SKIPPED'):
    print(s)
elif d.get('statusCode') == 500:
    print('ENV_ERROR')
else:
    print(d.get('errorMessage', 'UNKNOWN'))
" 2>/dev/null || echo "PARSE_ERROR")
  rm -f "$CANARY_OUT"

  if [ "$CANARY_STATUS" != "OK" ] && [ "$CANARY_STATUS" != "SKIPPED" ]; then
    echo "  ERROR: Canary returned status '$CANARY_STATUS' — auto-rolling back!"
    bash "$(dirname "$0")/rollback.sh"
    # Independent-channel surveillance per ROADMAP L221 — the 2-day
    # silent rollback chain (alpha-engine-data #274 retrospective)
    # showed the GitHub Actions red-icon is not load-bearing.
    # ``dedup_key`` collapses an image-wide rebuild that breaks N
    # Lambdas' canaries within the hour into one alert per (Lambda,
    # version) — lib v0.24.0 substrate (L221 retrofit 2026-05-22).
    # Best-effort; ``|| true`` never overrides this script's
    # ``exit 1``. Lib alerts CLI exits 0 if any channel (SNS or
    # Telegram) succeeded. Target is ``krepis.alerts`` (config#1339): the
    # alerts module relocated to krepis (MIT) at nousergon-lib v0.66.0 and
    # ``alpha_engine_lib.alerts`` is now a runpy-silent alias shim, so
    # ``-m alpha_engine_lib.alerts`` would no-op. krepis is pulled
    # transitively by the nousergon-lib pin (hard dep ``krepis>=0.2.0``).
    python3 -m krepis.alerts publish \
      --severity error \
      --source "alpha-engine-research/infrastructure/deploy.sh" \
      --dedup-key "canary-fail-${FUNCTION_MAIN}-v${VERSION}" \
      --message "Canary rolled back: ${FUNCTION_MAIN} canary returned status='${CANARY_STATUS}' — live alias reverted to prior version. See GitHub Actions log for full canary payload." \
      || true
    exit 1
  fi
  echo "  Canary passed (status=$CANARY_STATUS)"
}

# ── Alerts function: container image deployment ───────────────────────────────

build_and_deploy_alerts() {
  echo "=== Building container image for $FUNCTION_ALERTS ==="

  ECR_REPO_ALERTS="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_ALERTS}"

  # alpha-engine-lib is installed inside Dockerfile.alerts via pip from
  # public git+https (lib was flipped public 2026-05-03). No vendor
  # staging needed.

  # Build Docker image
  echo "Building Docker image..."
  docker build --platform linux/amd64 --provenance=false \
    -f Dockerfile.alerts \
    -t "$FUNCTION_ALERTS:latest" .

  # Authenticate with ECR
  echo "Authenticating with ECR..."
  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

  # Ensure ECR repository exists
  aws ecr describe-repositories --repository-names "$FUNCTION_ALERTS" --region "$REGION" &>/dev/null || \
    aws ecr create-repository --repository-name "$FUNCTION_ALERTS" --region "$REGION" > /dev/null

  # Tag and push
  echo "Pushing image to ECR..."
  docker tag "$FUNCTION_ALERTS:latest" "$ECR_REPO_ALERTS:latest"
  docker push "$ECR_REPO_ALERTS:latest"
  IMAGE_URI="$ECR_REPO_ALERTS:latest"

  echo "Deploying $FUNCTION_ALERTS..."

  # Build env var args

  if _lambda_function_exists "$FUNCTION_ALERTS"; then
    EXISTING_PKG=$(aws lambda get-function-configuration \
      --function-name "$FUNCTION_ALERTS" --region "$REGION" \
      --query "PackageType" --output text 2>/dev/null || echo "Zip")

    if [ "$EXISTING_PKG" = "Image" ]; then
      aws lambda update-function-code \
        --function-name "$FUNCTION_ALERTS" \
        --image-uri "$IMAGE_URI" \
        --region "$REGION" > /dev/null
    else
      # Zip → Image migration
      echo "  Migrating from zip to container image..."
      aws lambda delete-function --function-name "$FUNCTION_ALERTS" --region "$REGION"
      sleep 2
      aws lambda create-function \
        --function-name "$FUNCTION_ALERTS" \
        --package-type Image \
        --code "ImageUri=$IMAGE_URI" \
        --role "$ROLE_ARN" \
        --timeout 60 \
        --memory-size 256 \
        --region "$REGION" > /dev/null
      echo "  NOTE: EventBridge triggers were removed. Re-run setup-eventbridge.sh to restore."
    fi
  else
    aws lambda create-function \
      --function-name "$FUNCTION_ALERTS" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --role "$ROLE_ARN" \
      --timeout 60 \
      --memory-size 256 \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_ALERTS deployed (container image)."
}

# ── Eval-judge function: reuses the main container image ─────────────────────
#
# The eval-judge Lambda runs ``lambda/eval_judge_handler.py`` (LLM-as-judge
# orchestrator). It needs the same dependency set as the main runner
# (langchain_anthropic, alpha_engine_lib, prompt loader, schemas), so
# rather than build a parallel image we point this function at the same
# ECR image and override CMD via ``--image-config`` to
# ``eval_judge_handler.handler``.
#
# Prerequisite: build_and_deploy_main must have run at least once on this
# branch so the ECR ${ECR_REPO}:latest image contains
# /var/task/eval_judge_handler.py (Dockerfile COPY of lambda/eval_judge_handler.py).

deploy_eval_judge() {
  echo "=== Deploying $FUNCTION_EVAL_JUDGE (image-share with $FUNCTION_MAIN) ==="

  IMAGE_URI="$ECR_REPO:latest"
  IMAGE_CONFIG='{"Command":["eval_judge_handler.handler"]}'


  if _lambda_function_exists "$FUNCTION_EVAL_JUDGE"; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_EVAL_JUDGE" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$FUNCTION_EVAL_JUDGE" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_EVAL_JUDGE" \
      --image-config "$IMAGE_CONFIG" \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_EVAL_JUDGE" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --image-config "$IMAGE_CONFIG" \
      --role "$ROLE_ARN" \
      --timeout 900 \
      --memory-size 1024 \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_EVAL_JUDGE deployed (CMD=eval_judge_handler.handler)."

  echo "  Publishing Lambda version..."
  aws lambda wait function-updated --function-name "$FUNCTION_EVAL_JUDGE" --region "$REGION" 2>/dev/null || sleep 5
  VERSION=$(aws lambda publish-version \
    --function-name "$FUNCTION_EVAL_JUDGE" \
    --query "Version" --output text \
    --region "$REGION")
  echo "  Published version: $VERSION"
  aws lambda update-alias \
    --function-name "$FUNCTION_EVAL_JUDGE" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "$FUNCTION_EVAL_JUDGE" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION"
  echo "  Alias 'live' → version $VERSION"
}

# ── Eval-rolling-mean function: reuses the main container image ──────────────
#
# Rolling-4-week-mean derived metric Lambda (PR 4b). Same image-share
# pattern as eval_judge — overrides CMD to
# ``eval_rolling_mean_handler.handler`` at deploy time so the function
# runs that handler instead of handler.handler. The SNS alarm on the
# emitted eval metrics is codified in infrastructure/setup_eval_alarms.sh
# (L4578e — quality-floor + control-breach alarms, idempotent).

deploy_eval_rolling_mean() {
  echo "=== Deploying $FUNCTION_EVAL_ROLLING_MEAN (image-share with $FUNCTION_MAIN) ==="

  IMAGE_URI="$ECR_REPO:latest"
  IMAGE_CONFIG='{"Command":["eval_rolling_mean_handler.handler"]}'


  if _lambda_function_exists "$FUNCTION_EVAL_ROLLING_MEAN"; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$FUNCTION_EVAL_ROLLING_MEAN" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
      --image-config "$IMAGE_CONFIG" \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --image-config "$IMAGE_CONFIG" \
      --role "$ROLE_ARN" \
      --timeout 300 \
      --memory-size 512 \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_EVAL_ROLLING_MEAN deployed (CMD=eval_rolling_mean_handler.handler)."

  echo "  Publishing Lambda version..."
  aws lambda wait function-updated --function-name "$FUNCTION_EVAL_ROLLING_MEAN" --region "$REGION" 2>/dev/null || sleep 5
  VERSION=$(aws lambda publish-version \
    --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
    --query "Version" --output text \
    --region "$REGION")
  echo "  Published version: $VERSION"
  aws lambda update-alias \
    --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION"
  echo "  Alias 'live' → version $VERSION"
}

# ── deploy_rationale_clustering ─────────────────────────────────────────────
#
# Cross-week rationale clustering Lambda — same image-share + CMD-override
# pattern as eval_judge / eval_rolling_mean. CMD overrides to
# ``rationale_clustering_handler.handler``. Trigger wiring (weekly
# EventBridge after eval-rolling-mean finishes) lands separately.

deploy_rationale_clustering() {
  echo "=== Deploying $FUNCTION_RATIONALE_CLUSTERING (image-share with $FUNCTION_MAIN) ==="

  IMAGE_URI="$ECR_REPO:latest"
  IMAGE_CONFIG='{"Command":["rationale_clustering_handler.handler"]}'


  if _lambda_function_exists "$FUNCTION_RATIONALE_CLUSTERING"; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$FUNCTION_RATIONALE_CLUSTERING" --region "$REGION" 2>/dev/null || sleep 5
    # Bump timeout 600s → 900s (Lambda max) to absorb corpus growth.
    # Closes 5/23-SF P0 (a) — the 2026-05-24 trading-day-fix recovery
    # hit the 600s ceiling at event 269. Setting timeout on EVERY
    # update (not just create) so existing Lambdas pick up the bump
    # without a destroy-recreate cycle.
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --image-config "$IMAGE_CONFIG" \
      --timeout 900 \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --image-config "$IMAGE_CONFIG" \
      --role "$ROLE_ARN" \
      --timeout 900 \
      --memory-size 1024 \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_RATIONALE_CLUSTERING deployed (CMD=rationale_clustering_handler.handler)."

  echo "  Publishing Lambda version..."
  aws lambda wait function-updated --function-name "$FUNCTION_RATIONALE_CLUSTERING" --region "$REGION" 2>/dev/null || sleep 5
  VERSION=$(aws lambda publish-version \
    --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
    --query "Version" --output text \
    --region "$REGION")
  echo "  Published version: $VERSION"
  aws lambda update-alias \
    --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION"
  echo "  Alias 'live' → version $VERSION"
}

# ── Eval-judge batch chain: image-share + per-Lambda CMD override ───────────
#
# Three Lambdas share the main ECR image, each with a different CMD
# pointing at one of the three batch-chain handlers
# (eval_judge_{submit,poll,process}_handler.handler). Per-Lambda
# memory + timeout chosen for the workload:
#   * Submit  — plan-build + manifest write + one batch-create call.
#               Network-bound, no LLM. 512MB / 300s.
#   * Poll    — single retrieve API call. Trivial. 256MB / 60s.
#   * Process — streams all batch results + parses + persists +
#               sync Sonnet escalation tail. 1024MB / 900s
#               (the legacy single-Lambda's spec — bounded only by
#               the synchronous escalation tail for borderline Haiku
#               results, which is the same workload the legacy
#               single-Lambda ran).
#
# Prerequisite: build_and_deploy_main must have run at least once on
# this branch so the ECR ${ECR_REPO}:latest image contains
# /var/task/eval_judge_{submit,poll,process}_handler.py (Dockerfile
# COPY of lambda/eval_judge_{...}_handler.py).

_deploy_image_shared_lambda() {
  local fn_name="$1"
  local handler_module="$2"
  local timeout_s="$3"
  local memory_mb="$4"

  echo "=== Deploying $fn_name (image-share with $FUNCTION_MAIN) ==="

  local IMAGE_URI="$ECR_REPO:latest"
  local IMAGE_CONFIG
  IMAGE_CONFIG="{\"Command\":[\"${handler_module}.handler\"]}"


  if _lambda_function_exists "$fn_name"; then
    aws lambda update-function-code \
      --function-name "$fn_name" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$fn_name" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "$fn_name" \
      --image-config "$IMAGE_CONFIG" \
      --timeout "$timeout_s" \
      --memory-size "$memory_mb" \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$fn_name" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --image-config "$IMAGE_CONFIG" \
      --role "$ROLE_ARN" \
      --timeout "$timeout_s" \
      --memory-size "$memory_mb" \
      --region "$REGION" > /dev/null
  fi
  echo "  $fn_name deployed (CMD=${handler_module}.handler timeout=${timeout_s}s memory=${memory_mb}MB)."

  echo "  Publishing Lambda version..."
  aws lambda wait function-updated --function-name "$fn_name" --region "$REGION" 2>/dev/null || sleep 5
  local VERSION
  VERSION=$(aws lambda publish-version \
    --function-name "$fn_name" \
    --query "Version" --output text \
    --region "$REGION")
  echo "  Published version: $VERSION"
  aws lambda update-alias \
    --function-name "$fn_name" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION" 2>/dev/null || \
  aws lambda create-alias \
    --function-name "$fn_name" \
    --name live \
    --function-version "$VERSION" \
    --region "$REGION"
  echo "  Alias 'live' → version $VERSION"
}

deploy_eval_judge_batch() {
  _deploy_image_shared_lambda "$FUNCTION_EVAL_JUDGE_SUBMIT"  "eval_judge_submit_handler"  300 512
  _deploy_image_shared_lambda "$FUNCTION_EVAL_JUDGE_POLL"    "eval_judge_poll_handler"     60 256
  _deploy_image_shared_lambda "$FUNCTION_EVAL_JUDGE_PROCESS" "eval_judge_process_handler" 900 1024
}

# Daily cost aggregation Lambda — ROADMAP L1146. Shared image with the
# main runner; CMD override sets the entry point. Timeout 300s (5min)
# is comfortable for the ~minutes-of-S3-reads on a Saturday's _cost_raw
# partition (~thousands of JSONL files × small parquet write).
deploy_aggregate_costs() {
  _deploy_image_shared_lambda "$FUNCTION_AGGREGATE_COSTS" "aggregate_costs_handler" 300 512
}

# Standalone scanner Lambda — ROADMAP L1995 Phase 1. Shared image with
# the main runner; CMD override sets the entry point. Timeout 300s
# (5min) covers feature-store read + ~903-ticker quant filter pass +
# S3 write — pure compute, no LLM calls. Memory 1024MB matches the
# main runner's headroom for ArcticDB / pandas working sets.
#
# The CloudWatch metric filter + degradation alarm on the scanner's
# candidate count (config#785) is codified in
# infrastructure/setup_scanner_alarm.sh (idempotent; run once after the
# first scanner deploy creates the log group), mirroring the eval alarms
# in setup_eval_alarms.sh.
deploy_scanner() {
  _deploy_image_shared_lambda "$FUNCTION_SCANNER" "scanner_handler" 300 1024
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

case "$TARGET" in
  main)                  build_and_deploy_main ;;
  alerts)                build_and_deploy_alerts ;;  # ci-deploy-guard: manual — alerts Lambda deployed on demand, not on every merge
  eval_judge)            deploy_eval_judge ;;
  eval_judge_batch)      deploy_eval_judge_batch ;;
  eval_rolling_mean)     deploy_eval_rolling_mean ;;
  rationale_clustering)  deploy_rationale_clustering ;;
  aggregate_costs)       deploy_aggregate_costs ;;
  scanner)               deploy_scanner ;;
  both)                  build_and_deploy_main; build_and_deploy_alerts ;;  # ci-deploy-guard: manual — aggregate convenience target
  all)                   build_and_deploy_main; build_and_deploy_alerts; deploy_eval_judge; deploy_eval_judge_batch; deploy_eval_rolling_mean; deploy_rationale_clustering; deploy_aggregate_costs; deploy_scanner ;;  # ci-deploy-guard: manual — aggregate convenience target
  *)                     echo "Usage: $0 [main|alerts|eval_judge|eval_judge_batch|eval_rolling_mean|rationale_clustering|aggregate_costs|scanner|both|all]"; exit 1 ;;
esac

echo ""
echo "Deployment complete."
echo ""
