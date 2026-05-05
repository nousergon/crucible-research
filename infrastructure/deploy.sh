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
FUNCTION_EVAL_ROLLING_MEAN="alpha-engine-research-eval-rolling-mean"
FUNCTION_RATIONALE_CLUSTERING="alpha-engine-research-rationale-clustering"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="alpha-engine-research"
BUILD_DIR="lambda/package"

# ECR repository for container image deployment
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || echo "ACCOUNT_ID")
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/alpha-engine-research-role}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_MAIN}"

TARGET="${1:-both}"

# ── Lambda env vars from lambda.env ──────────────────────────────────────────
# Reads lambda.env (gitignored) and builds the JSON for --environment.

# Master .env lives in alpha-engine-data; fall back to local .env
LAMBDA_ENV_FILE="$(dirname "$(pwd)")/alpha-engine-data/.env"
if [ ! -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_FILE=".env"
fi

build_lambda_env_json() {
  if [ ! -f "$LAMBDA_ENV_FILE" ]; then
    echo "WARNING: $LAMBDA_ENV_FILE not found — Lambda will have no env vars configured." >&2
    echo ""
    return
  fi
  # Parse KEY=VALUE lines from .env, stopping at LAMBDA_SKIP marker.
  # Vars after LAMBDA_SKIP are local-only (AWS creds etc).
  python3 -c "
import json
env = {}
with open('$LAMBDA_ENV_FILE') as f:
    for line in f:
        line = line.strip()
        if line == '# LAMBDA_SKIP':
            break
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        # Strip surrounding quotes (single or double)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):
            val = val[1:-1]
        if key and val:
            env[key] = val
if env:
    print(json.dumps({'Variables': env}))
else:
    print('')
"
}

LAMBDA_ENV_JSON=$(build_lambda_env_json)

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
      echo "Hint: clone cipher813/alpha-engine-config as a sibling directory,"
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
        echo "Hint: clone cipher813/alpha-engine-config as a sibling directory,"
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
      echo "Hint: clone cipher813/alpha-engine-config as a sibling directory,"
      echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
      exit 1
    fi
  fi

  # Build Docker image
  echo "Building Docker image..."
  docker build --platform linux/amd64 --provenance=false -t "$FUNCTION_MAIN:latest" .

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

  # Build env var args from lambda.env
  ENV_ARGS=()
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    ENV_ARGS=(--environment "$LAMBDA_ENV_JSON")
    echo "  Env vars from lambda.env: $(echo "$LAMBDA_ENV_JSON" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin).get('Variables',{}).keys()))")"
  fi

  if aws lambda get-function --function-name "$FUNCTION_MAIN" --region "$REGION" &>/dev/null; then
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
      # Sync env vars from lambda.env
      if [ -n "$LAMBDA_ENV_JSON" ]; then
        echo "  Waiting for code update to complete..."
        aws lambda wait function-updated --function-name "$FUNCTION_MAIN" --region "$REGION" 2>/dev/null || sleep 5
        aws lambda update-function-configuration \
          --function-name "$FUNCTION_MAIN" \
          --environment "$LAMBDA_ENV_JSON" \
          --region "$REGION" > /dev/null
      fi
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
        "${ENV_ARGS[@]}" \
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
      "${ENV_ARGS[@]}" \
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
  aws lambda invoke \
    --function-name "${FUNCTION_MAIN}:live" \
    --payload '{"dry_run_llm": true}' \
    --cli-binary-format raw-in-base64-out \
    --region "$REGION" \
    "$CANARY_OUT" > /dev/null

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
  ENV_ARGS=()
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    ENV_ARGS=(--environment "$LAMBDA_ENV_JSON")
  fi

  if aws lambda get-function --function-name "$FUNCTION_ALERTS" --region "$REGION" &>/dev/null; then
    EXISTING_PKG=$(aws lambda get-function-configuration \
      --function-name "$FUNCTION_ALERTS" --region "$REGION" \
      --query "PackageType" --output text 2>/dev/null || echo "Zip")

    if [ "$EXISTING_PKG" = "Image" ]; then
      aws lambda update-function-code \
        --function-name "$FUNCTION_ALERTS" \
        --image-uri "$IMAGE_URI" \
        --region "$REGION" > /dev/null
      if [ -n "$LAMBDA_ENV_JSON" ]; then
        echo "  Waiting for code update to complete..."
        aws lambda wait function-updated --function-name "$FUNCTION_ALERTS" --region "$REGION" 2>/dev/null || sleep 5
        aws lambda update-function-configuration \
          --function-name "$FUNCTION_ALERTS" \
          --environment "$LAMBDA_ENV_JSON" \
          --region "$REGION" > /dev/null
      fi
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
        "${ENV_ARGS[@]}" \
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
      "${ENV_ARGS[@]}" \
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

  ENV_ARGS=()
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    ENV_ARGS=(--environment "$LAMBDA_ENV_JSON")
  fi

  if aws lambda get-function --function-name "$FUNCTION_EVAL_JUDGE" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_EVAL_JUDGE" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$FUNCTION_EVAL_JUDGE" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_EVAL_JUDGE" \
      --image-config "$IMAGE_CONFIG" \
      "${ENV_ARGS[@]}" \
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
      "${ENV_ARGS[@]}" \
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
# runs that handler instead of handler.handler. Trigger wiring (weekly
# EventBridge rule + SNS alarm) lands in PR 4c.

deploy_eval_rolling_mean() {
  echo "=== Deploying $FUNCTION_EVAL_ROLLING_MEAN (image-share with $FUNCTION_MAIN) ==="

  IMAGE_URI="$ECR_REPO:latest"
  IMAGE_CONFIG='{"Command":["eval_rolling_mean_handler.handler"]}'

  ENV_ARGS=()
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    ENV_ARGS=(--environment "$LAMBDA_ENV_JSON")
  fi

  if aws lambda get-function --function-name "$FUNCTION_EVAL_ROLLING_MEAN" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$FUNCTION_EVAL_ROLLING_MEAN" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_EVAL_ROLLING_MEAN" \
      --image-config "$IMAGE_CONFIG" \
      "${ENV_ARGS[@]}" \
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
      "${ENV_ARGS[@]}" \
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

  ENV_ARGS=()
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    ENV_ARGS=(--environment "$LAMBDA_ENV_JSON")
  fi

  if aws lambda get-function --function-name "$FUNCTION_RATIONALE_CLUSTERING" --region "$REGION" &>/dev/null; then
    aws lambda update-function-code \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --image-uri "$IMAGE_URI" \
      --region "$REGION" > /dev/null
    echo "  Waiting for code update to complete..."
    aws lambda wait function-updated --function-name "$FUNCTION_RATIONALE_CLUSTERING" --region "$REGION" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --image-config "$IMAGE_CONFIG" \
      "${ENV_ARGS[@]}" \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --image-config "$IMAGE_CONFIG" \
      --role "$ROLE_ARN" \
      --timeout 600 \
      --memory-size 1024 \
      "${ENV_ARGS[@]}" \
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

# ── Dispatch ─────────────────────────────────────────────────────────────────

case "$TARGET" in
  main)                  build_and_deploy_main ;;
  alerts)                build_and_deploy_alerts ;;
  eval_judge)            deploy_eval_judge ;;
  eval_rolling_mean)     deploy_eval_rolling_mean ;;
  rationale_clustering)  deploy_rationale_clustering ;;
  both)                  build_and_deploy_main; build_and_deploy_alerts ;;
  all)                   build_and_deploy_main; build_and_deploy_alerts; deploy_eval_judge; deploy_eval_rolling_mean; deploy_rationale_clustering ;;
  *)                     echo "Usage: $0 [main|alerts|eval_judge|eval_rolling_mean|rationale_clustering|both|all]"; exit 1 ;;
esac

echo ""
echo "Deployment complete."
echo ""
