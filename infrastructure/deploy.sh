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
FUNCTION_SIGNALS_ENVELOPE="alpha-engine-research-signals-envelope"
FUNCTION_OPENROUTER_SHADOW="alpha-engine-research-openrouter-shadow"
FUNCTION_PERTURBATION_BATTERY="alpha-engine-research-perturbation-battery"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="alpha-engine-research"
BUILD_DIR="lambda/package"

# ECR repository for container image deployment
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION" 2>/dev/null || echo "ACCOUNT_ID")
ROLE_ARN="${LAMBDA_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/alpha-engine-research-role}"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${FUNCTION_MAIN}"

TARGET="${1:-both}"

# ── Deploy stamp: what commit is this function actually running? ─────────────
#
# alpha-engine-config-I7840. Until now the only stamp was the Docker ENV
# ``ALPHA_ENGINE_CODE_SHA`` baked by ``--build-arg GIT_SHA`` (Dockerfile L69).
# Runtime code can read it, but nothing OUTSIDE the function can: an image ENV
# is not surfaced by ``lambda get-function-configuration``, so answering "is
# this Lambda behind main?" required invoking it. That is why the 2026-08-18
# scanner ran two days of pre-fix code with nobody able to see it (I7645 →
# I7819 → I7840): the fact existed and was unreadable.
#
# Setting the SAME name as a function-level environment variable makes it
# readable without a redeploy and without an invoke. Same name deliberately —
# a second stamp key would be a fork, and the two would diverge in exactly the
# direction that produces a confident wrong answer. The function-level value
# overrides the image ENV with an identical value for the main image, and
# supplies one for the image-shared Lambdas and for the alerts image, which
# has no GIT_SHA build-arg at all.
#
# Resolved ONCE here, at global scope, rather than inside build_and_deploy_main:
# deploy.yml invokes this script once PER TARGET, so `deploy.sh scanner` never
# enters the main build path and would otherwise stamp nothing.
GIT_SHA="${GITHUB_SHA:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse HEAD 2>/dev/null || echo '')}"
if [[ ! "$GIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: could not resolve a 40-hex commit SHA to stamp this deploy with." >&2
  echo "  GITHUB_SHA='${GITHUB_SHA:-<unset>}' and 'git rev-parse HEAD' gave '${GIT_SHA}'." >&2
  echo "  Refusing to publish an UNSTAMPED function: an unstamped deploy is exactly" >&2
  echo "  the invisible state alpha-engine-config-I7840 exists to remove, and the" >&2
  echo "  drift detector would report it as unmeasured forever." >&2
  exit 1
fi

# A deploy built from a working tree with uncommitted changes carries a SHA
# that does NOT describe the running code. Stamping the SHA alone would make
# the drift detector report `in_sync` about code that exists nowhere in git.
# Say so instead. Empty `git status --porcelain` output means clean.
if [ -n "$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." status --porcelain 2>/dev/null)" ]; then
  GIT_TREE_DIRTY="true"
  echo "WARNING: deploying from a DIRTY working tree — functions will be stamped DEPLOY_STAMP_DIRTY=true."
else
  GIT_TREE_DIRTY="false"
fi


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

# ── Post-deploy alias-propagation verification ────────────────────────────
#
# config#2766: a Saturday production run reproduced the exact pre-fix
# eval-rolling-mean zero-sigma OUT_OF_CONTROL output days after the fix
# (crucible-research#417) merged, with deploy.yml reporting success on
# every run since. The invoked ARN is the 'live' alias (see
# infrastructure/step_function_advisory.json), not $LATEST — every deploy
# function below does `update-function-code` -> `publish-version` ->
# `update-alias`, but nothing ever confirmed the alias actually ended up
# pointing at the code that was just published. The `update-alias ... ||
# create-alias` fallback also swallows update-alias's stderr unconditionally,
# so a transient AWS-side failure there would silently fall through instead
# of surfacing. Comparing CodeSha256 (Lambda's own content hash, already
# used identically by both qualifiers) directly verifies the one invariant
# every invoker of these Lambdas depends on, regardless of which upstream
# mechanism (ECR :latest tag race, transient alias-update failure, control-
# plane eventual consistency) would otherwise have caused the drift.
_verify_live_alias() {
  local fn_name="$1"
  local expected_version="$2"
  local latest_sha alias_sha

  latest_sha=$(aws lambda get-function --function-name "$fn_name" \
    --qualifier '$LATEST' --query 'Configuration.CodeSha256' \
    --output text --region "$REGION")
  alias_sha=$(aws lambda get-function --function-name "$fn_name" \
    --qualifier live --query 'Configuration.CodeSha256' \
    --output text --region "$REGION")

  if [[ "$latest_sha" != "$alias_sha" ]]; then
    echo "ERROR: $fn_name alias 'live' (CodeSha256=$alias_sha) does not match" >&2
    echo "  \$LATEST (CodeSha256=$latest_sha) after publishing version $expected_version." >&2
    echo "  The deploy did NOT propagate to the invoked alias -- failing the" >&2
    echo "  build instead of leaving stale code live behind a green CI run" >&2
    echo "  (config#2766)." >&2
    exit 1
  fi
  echo "  Verified: alias 'live' -> version $expected_version, CodeSha256 $alias_sha matches \$LATEST."
}

# ── Throttle-aware Lambda invoke ─────────────────────────────────────────────
#
# The bounded, jittered "retry ONLY on the throttle/concurrency signal" invoke
# used to live here as an inline Bash helper; it is now the shared
# ``krepis.aws invoke-canary`` CLI (config#1494) so all four
# deploy.sh consumers use ONE implementation of the throttle idiom instead of a
# per-repo Bash copy (the copies drifted; the research one bit CI 2026-07-01).
#
# Contract (``python3 -m krepis.aws invoke-canary``):
#   --function-name NAME:alias  --payload 'JSON'  --out FILE  [--region R]
#   [--max-attempts N (default 6)]  [--label L]
#   → writes the response payload bytes to FILE, prints the invoke METADATA
#     JSON ``{"StatusCode","FunctionError","ExecutedVersion"}`` to stdout,
#     exits 0 on invoke-API success, exits 1 on a non-throttle boto error or
#     throttle/concurrency exhaustion. Reserved-concurrency=1 singleton guard
#     (the research runner has ONE slot fleet-wide) can have its lone slot
#     legitimately busy when the canary fires; the CLI retries ONLY on
#     TooManyRequestsException / ReservedFunctionConcurrentInvocationLimitExceeded
#     with exponential backoff + jitter, and fails loud on exhaustion.
# NOTE: boto3 path — no ``--cli-binary-format``/base64 dance.

# ── Main function: container image deployment ────────────────────────────────

# -- Router addressing for the research Lambda (config-I6367 / I6373) -------
#
# Brian's ruling 2026-08-03: no agent may be directly linked to OpenRouter.
# `producers/single_agent.py` addresses the `high` model GROUP through the
# authenticated router edge, and needs six facts it cannot derive for itself.
#
# MERGE, never replace. `update-function-configuration --environment` REPLACES
# the whole variable map, and this function's other ~20 variables (provider
# keys, RAG_DATABASE_URL, LangSmith config) are NOT codified anywhere in this
# repo -- they exist only on the live function. Writing a fresh map here would
# delete every one of them. That gap is tracked as its own issue; until it is
# closed, the only safe shape is read-modify-write.
#
# Nothing is echoed. The current map carries live credentials, so it is read
# into a file and merged by python, never printed, never passed on a command
# line. Do NOT trace this script (see AGENTS.md: a shell trace expands the
# variable JSON).
# -- Cost-sink addressing for EVERY research Lambda (config-I7179) ----------
#
# Measured 2026-08-13: every per-call cost record in
# decision_artifacts/_cost_raw/ came from ONE producer -- the Think Tank,
# which left the weekly pipeline on 2026-08-10 -- while every LLM-calling
# stage of that pipeline was attributed to nothing. The cause was not a
# broken writer: `krepis.llm.LLMClient` only emitted when a call site
# remembered to pass `cost_sink=`, so coverage equalled the set of authors
# who thought about it, and decayed with every stage added.
#
# krepis>=0.57.0 inverts that: a client with no `cost_sink` argument
# resolves one from KREPIS_COST_SINK_BUCKET + KREPIS_COST_SINK_PREFIX.
# Setting them here makes emission a property of the DEPLOYMENT, so a new
# Lambda -- or a new LLM call inside an existing one -- emits by
# construction and cannot silently reproduce the gap.
#
# Applied to EVERY function this script publishes, not to a hand-picked
# subset. A function that makes no LLM call simply never emits; picking
# the subset by hand is the same judgement that produced the gap.
#
# ORDERING IS LOAD-BEARING: a published Lambda version snapshots the
# environment, and the `live` alias points at a published version. This
# must run BEFORE `publish-version` or the alias serves a version without
# the variables, while `get-function-configuration` on $LATEST shows them
# set -- a deploy-path defect that is invisible in every file.
#
# Merge, never replace: `krepis.aws merge-lambda-env` is read-modify-write
# and preserves the provider keys, RAG_DATABASE_URL and LangSmith config
# that exist only on the live functions. It echoes key NAMES only.
_apply_cost_sink_env() {
  local fn="$1"
  echo "  Applying cost-sink addressing to $fn (merge, not replace)..."
  python3 -m krepis.aws merge-lambda-env --function-name "$fn" --set KREPIS_COST_SINK_BUCKET="$BUCKET" --set KREPIS_COST_SINK_PREFIX=decision_artifacts/_cost_raw --region "$REGION"
}

# ── Deploy stamp applier (alpha-engine-config-I7840) ─────────────────────────
#
# Merge, never replace — same read-modify-write CLI as the cost-sink applier,
# so provider keys, RAG_DATABASE_URL and LangSmith config that exist only on
# the live functions survive.
#
# ORDERING IS LOAD-BEARING, twice over:
#   1. This must run BEFORE `publish-version`, or the `live` alias serves a
#      version whose environment has no stamp while `$LATEST` shows one — the
#      deploy-path defect that is invisible in every file (config#2766), and
#      the drift detector reads the ALIAS, not $LATEST.
#   2. It must WAIT first. `_apply_cost_sink_env` runs immediately before it
#      and leaves the function in LastUpdateStatus=InProgress for a few
#      seconds; a configuration update issued inside that window fails with
#      ResourceConflictException, which under `set -euo pipefail` aborts the
#      whole deploy (observed live on the router-env applier, deploy runs
#      30866612804 / 30867141385, 2026-08-04).
_apply_deploy_stamp_env() {
  local fn="$1"
  echo "  Stamping $fn with ALPHA_ENGINE_CODE_SHA=$GIT_SHA (dirty=$GIT_TREE_DIRTY, merge not replace)..."
  aws lambda wait function-updated --function-name "$fn" --region "$REGION" 2>/dev/null || sleep 5
  python3 -m krepis.aws merge-lambda-env --function-name "$fn" --set ALPHA_ENGINE_CODE_SHA="$GIT_SHA" --set DEPLOY_STAMP_DIRTY="$GIT_TREE_DIRTY" --region "$REGION"
}

_apply_router_env() {
  local fn="$1"
  echo "  Applying router addressing to $fn (merge, not replace)..."

  # Lambda serializes updates per function. `update-function-code` leaves the
  # function in LastUpdateStatus=InProgress for a few seconds, and a
  # configuration update issued inside that window fails with
  #
  #   ResourceConflictException: The operation cannot be performed at this
  #   time. An update is in progress for resource: ...
  #
  # which under `set -euo pipefail` aborts the whole deploy. Observed live on
  # the first two runs of this function (crucible-research deploy runs
  # 30866612804 and 30867141385, 2026-08-04) -- the merge itself computed
  # correctly and only the write raced. Every other update in this script
  # already waits first; this one has to as well.
  aws lambda wait function-updated --function-name "$fn" --region "$REGION" 2>/dev/null || sleep 5

  local tmp_cur tmp_new
  tmp_cur="$(mktemp)"; tmp_new="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '$tmp_cur' '$tmp_new'" RETURN

  aws lambda get-function-configuration \
    --function-name "$fn" --region "$REGION" \
    --query "Environment.Variables" --output json > "$tmp_cur" 2>/dev/null \
    || echo '{}' > "$tmp_cur"

  ROUTER_URL="${ROUTER_URL:-https://router.nousergon.ai:8443}" \
  ROUTER_CREDENTIAL_SECRET="${ROUTER_CREDENTIAL_SECRET:-ROUTER_CONSUMER_RESEARCH}" \
  python3 - "$tmp_cur" "$tmp_new" <<'PYEOF'
import json, os, sys

cur_path, new_path = sys.argv[1], sys.argv[2]
with open(cur_path) as fh:
    variables = json.load(fh) or {}

# exec_context names WHERE CODE RUNS (model-router-policy R28), never how it
# is attached and never which routes are wanted. The registry declares
# `lambda` on NO model entry, deliberately -- a Lambda has no local egress
# proxy and no private-network peer, so the router is its only path and this
# call site FAILS CLOSED rather than reaching a provider endpoint unscanned.
variables.update({
    "KREPIS_EXEC_CONTEXT": "lambda",
    "KREPIS_LITELLM_PROXY_URL": os.environ["ROUTER_URL"],
    # Its OWN secret, not LITELLM_MASTER_KEY: the edge identifies a consumer
    # BY its credential VALUE and krepis.secrets resolves SSM BEFORE
    # os.environ, so a shared name collapses this Lambda into the director's
    # identity at the edge however the environment is set.
    "KREPIS_ROUTER_CREDENTIAL_SECRET": os.environ["ROUTER_CREDENTIAL_SECRET"],
    # crucible-research is PUBLIC, so private-docs/LLM_MODEL_REGISTRY.yaml is
    # correctly absent from the image. All three are required: krepis'
    # AppConfig path is opt-in on the application id and SWALLOWS its own
    # errors, falling through to a filesystem walk that finds nothing here --
    # so a missing one surfaces later as "LLM_MODEL_REGISTRY.yaml not found",
    # naming neither AppConfig nor the cause.
    "KREPIS_APPCONFIG_APPLICATION": "alpha-engine",
    "KREPIS_APPCONFIG_CONFIG_PROFILE": "llm-model-registry",
    "KREPIS_APPCONFIG_ENVIRONMENT": "production",
})

with open(new_path, "w") as fh:
    json.dump({"Variables": variables}, fh)
print(f"    merged 6 router variables into {len(variables)} total (values not shown)")
PYEOF

  aws lambda update-function-configuration \
    --function-name "$fn" \
    --environment "file://$tmp_new" \
    --region "$REGION" > /dev/null
  aws lambda wait function-updated --function-name "$fn" --region "$REGION" 2>/dev/null || sleep 5
  echo "    router addressing applied."
}

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

  # Resolve the research module config source: experiment-package FIRST, legacy
  # top-level dir as fallback (config#1042 experiment-package adoption — matches
  # the executor config_loader / backtester spot_backtest.sh precedent). The
  # package copy is canonical; the legacy `research/` dir is being removed once
  # every reader is package-aware — this was the last one. $ALPHA_ENGINE_EXPERIMENT_ID
  # is not injected in the GitHub-Actions Lambda-deploy context, so it defaults
  # to "reference" (the prod experiment). Fallback keeps this working through the
  # cutover window (legacy still present) AND after the legacy dir is deleted.
  EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"
  RESEARCH_CFG_DIR=""
  for candidate in \
    "$CONFIG_REPO_DIR/experiments/$EXPERIMENT_ID/research" \
    "$CONFIG_REPO_DIR/research"; do
    if [ -d "$candidate" ]; then
      RESEARCH_CFG_DIR="$candidate"
      break
    fi
  done
  if [ -z "$RESEARCH_CFG_DIR" ]; then
    echo "ERROR: research config dir not found — tried (package-first):"
    echo "  $CONFIG_REPO_DIR/experiments/$EXPERIMENT_ID/research/ (experiment package)"
    echo "  $CONFIG_REPO_DIR/research/ (legacy top-level)"
    echo "Hint: clone nousergon/alpha-engine-config as a sibling directory,"
    echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
    echo "      (and optionally ALPHA_ENGINE_EXPERIMENT_ID; default: reference)"
    exit 1
  fi
  echo "Research config source: $RESEARCH_CFG_DIR (experiment=$EXPERIMENT_ID)"

  # -- prompts -------------------------------------------------------------
  if [ -d "config/prompts" ] && ls config/prompts/*.txt &>/dev/null; then
    echo "Using existing config/prompts/ (local dev workflow)"
  else
    if [ -d "$RESEARCH_CFG_DIR/prompts" ]; then
      echo "Staging research prompts from $RESEARCH_CFG_DIR/prompts/..."
      mkdir -p config/prompts
      cp "$RESEARCH_CFG_DIR/prompts/"*.txt config/prompts/
      PROMPTS_STAGED_FROM_CONFIG_REPO=1
    else
      echo "ERROR: research prompts not found — tried:"
      echo "  config/prompts/ (local dev)"
      echo "  $RESEARCH_CFG_DIR/prompts/ (config repo, experiment=$EXPERIMENT_ID)"
      echo "Hint: clone nousergon/alpha-engine-config as a sibling directory,"
      echo "      or set CONFIG_REPO_DIR=/path/to/alpha-engine-config"
      exit 1
    fi
  fi

  # -- scoring.yaml + universe.yaml ---------------------------------------
  for yaml in scoring.yaml universe.yaml thinktank.yaml; do
    if [ -f "config/$yaml" ]; then
      echo "Using existing config/$yaml (local dev workflow)"
    else
      src="$RESEARCH_CFG_DIR/$yaml"
      if [ -f "$src" ]; then
        echo "Staging config/$yaml from $src..."
        cp "$src" "config/$yaml"
        YAMLS_STAGED_FROM_CONFIG_REPO+=("$yaml")
      else
        echo "ERROR: config/$yaml not found — tried:"
        echo "  config/$yaml (local dev)"
        echo "  $src (config repo, experiment=$EXPERIMENT_ID)"
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
  # GIT_SHA is resolved once at global scope (see the deploy-stamp block near
  # the top) so every target stamps the same value, not just this one.
  echo "  Stamping image with GIT_SHA=${GIT_SHA}"

  # Build Docker image
  echo "Building Docker image..."
  docker build --platform linux/amd64 --provenance=false \
    --build-arg "GIT_SHA=${GIT_SHA}" \
    --label "org.opencontainers.image.revision=${GIT_SHA}" \
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

  _apply_router_env "$FUNCTION_MAIN"
  _apply_cost_sink_env "$FUNCTION_MAIN"
  _apply_deploy_stamp_env "$FUNCTION_MAIN"

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
  _verify_live_alias "$FUNCTION_MAIN" "$VERSION"

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
  if ! python3 -m krepis.aws invoke-canary \
      --function-name "${FUNCTION_MAIN}:live" \
      --payload '{"dry_run_llm": true}' \
      --out "$CANARY_OUT" \
      --region "$REGION" \
      --max-attempts 6 \
      --label "${FUNCTION_MAIN}-canary" >/dev/null; then
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
  # On any non-pass, SURFACE THE REAL DETAIL rather than a bare 'UNKNOWN':
  # the handler's ERROR return carries the message under ``error`` (outer
  # except) — NOT ``errorMessage`` — and the 500 env path under ``body``;
  # reading only ``errorMessage`` collapsed every genuine canary failure to
  # an opaque 'UNKNOWN', making the rollback un-diagnosable (2026-07-21
  # incident). ``errorMessage`` is still consulted for a Lambda-runtime
  # unhandled crash, and the raw payload is the last-resort fallback.
  CANARY_STATUS=$(python3 -c "
import json, sys
d = json.load(open('$CANARY_OUT'))
s = d.get('status', '')
if s in ('OK', 'SKIPPED'):
    print(s)
else:
    detail = (d.get('error') or d.get('body') or d.get('errorMessage')
              or d.get('reason') or json.dumps(d))
    detail = ' '.join(str(detail).split())[:300]
    label = 'ENV_ERROR' if d.get('statusCode') == 500 else (s or 'UNKNOWN')
    print(f'{label}: {detail}')
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
    --label "org.opencontainers.image.revision=${GIT_SHA}" \
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
  _apply_deploy_stamp_env "$FUNCTION_ALERTS"
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

  _apply_cost_sink_env "$FUNCTION_EVAL_JUDGE"

  _apply_deploy_stamp_env "$FUNCTION_EVAL_JUDGE"

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
  _verify_live_alias "$FUNCTION_EVAL_JUDGE" "$VERSION"
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

  _apply_cost_sink_env "$FUNCTION_EVAL_ROLLING_MEAN"

  _apply_deploy_stamp_env "$FUNCTION_EVAL_ROLLING_MEAN"

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
  _verify_live_alias "$FUNCTION_EVAL_ROLLING_MEAN" "$VERSION"
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
    # Bump memory 1024 → 3008MB (config#1650 item 3): the SF Task
    # ceiling was already synced to the Lambda's 900s timeout
    # (nousergon-data#606), but the clustering pass still chronically
    # timed out — Lambda CPU scales with memory, so this Lambda was
    # CPU-starved for a compute-bound clustering workload. Set on
    # EVERY update (not just create), same rationale as the timeout
    # bump above: existing Lambdas must pick this up without a
    # destroy-recreate cycle.
    aws lambda update-function-configuration \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --image-config "$IMAGE_CONFIG" \
      --timeout 900 \
      --memory-size 3008 \
      --region "$REGION" > /dev/null
  else
    aws lambda create-function \
      --function-name "$FUNCTION_RATIONALE_CLUSTERING" \
      --package-type Image \
      --code "ImageUri=$IMAGE_URI" \
      --image-config "$IMAGE_CONFIG" \
      --role "$ROLE_ARN" \
      --timeout 900 \
      --memory-size 3008 \
      --region "$REGION" > /dev/null
  fi
  echo "  $FUNCTION_RATIONALE_CLUSTERING deployed (CMD=rationale_clustering_handler.handler)."

  _apply_cost_sink_env "$FUNCTION_RATIONALE_CLUSTERING"

  _apply_deploy_stamp_env "$FUNCTION_RATIONALE_CLUSTERING"

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
  _verify_live_alias "$FUNCTION_RATIONALE_CLUSTERING" "$VERSION"
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

  _apply_cost_sink_env "$fn_name"

  _apply_deploy_stamp_env "$fn_name"

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
  _verify_live_alias "$fn_name" "$VERSION"
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
# the main runner; CMD override sets the entry point. Covers feature-store
# read + ~903-ticker quant filter pass + universe board + membership +
# leaderboard + S3 writes — pure compute, no LLM calls. Memory 1024MB
# matches the main runner's headroom for ArcticDB / pandas working sets
# (observed Max Memory Used 520MB, so memory is not the binding budget).
#
# TIMEOUT 450s, sized to observed p95 × 1.5 per sf-pipeline-policy.md §4,
# NOT to a round number. Raised from 300s on 2026-08-11
# (alpha-engine-config-I6855) after both preopen attempts died at exactly
# 300.00s and the pipeline terminated DEGRADED, losing that day's universe
# board, membership, leaderboard and trajectory. Duration history over
# 7/28-8/11 put p95 at ~290s — 7/31 came within 3% of the ceiling and 8/6
# hit it once, two weeks of creep with nothing paging.
#
# Raising the ceiling is HALF the fix and is not a licence to absorb growth
# silently (§1.2). The other half is the union projection in
# data/fetchers/feature_store_reader.py, which removed one of the two
# whole-universe read_batch round-trips, and the duration alarm below,
# which pages at 70% of this value so the next creep is seen before it
# fails. If this number needs raising again, find out what grew first.
#
# The SF task budget must stay ABOVE this: a lambda:invoke whose
# timeoutSeconds is below the function's own timeout can never bind, and
# the SF sees Sandbox.Timedout instead of States.Timeout. Asserted by
# nousergon-data/tests/test_sf_lambda_timeout_ordering.py.
#
# The CloudWatch metric filter + degradation alarm on the scanner's
# candidate count (config#785) and the duration alarm (config-I6855) are
# codified in infrastructure/setup_scanner_alarm.sh (idempotent; run once
# after the first scanner deploy creates the log group), mirroring the eval
# alarms in setup_eval_alarms.sh.
deploy_scanner() {
  _deploy_image_shared_lambda "$FUNCTION_SCANNER" "scanner_handler" 450 1024
  # AFTER the update, never before: the duration alarm derives its threshold
  # from the LIVE timeout, so running it first would codify the old ceiling.
  # Not guarded — a timeout raised with no alarm behind it is the state that
  # produced I6855, and a deploy that silently half-applies is worse than a
  # red one.
  bash "$(dirname "${BASH_SOURCE[0]}")/setup_scanner_alarm.sh" duration
}

# Daily think-tank Lambda deploy target — RETIRED (alpha-engine-config-I5777,
# 2026-08-04). The §47 spot cutover (config-I5208 daily / config-I5758 weekly)
# repointed both invokers of `alpha-engine-research-thinktank` onto
# `alpha-engine-thinktank-spot-dispatcher`; measured 2026-08-04, the function
# has had zero invocations since 2026-07-29, carries no resource-based policy
# (no principal can invoke it), and no deployed state machine names it. This
# target published Docker images to a Lambda nothing calls, on every push, for
# no benefit. `lambda/thinktank_handler.py` is NOT dead — it is still imported
# and run in-process by `infrastructure/thinktank_box_runner.py` on the spot
# box, so its tests and source stay. Only the Lambda-specific publish path is
# gone. The AWS Lambda resource itself is not deleted by this change — see
# alpha-engine-config-I5777 for the operator-run deletion command.

# Signals-envelope Lambda — alpha-engine-config epic #2515 Phase B. Shared
# image with the main runner; CMD override sets the entry to
# signals_envelope_handler.handler. Invoked ONLY by the weekly SF's
# SignalsEnvelope state (arn:aws:states:::lambda:invoke, synchronous),
# placed immediately AFTER RegimeSubstrate so the regime read this producer
# takes is same-day fresh (config#1580's no-week-old-data invariant) — never
# triggered by EventBridge directly. Timeout 300s is generous: the envelope
# is a read-two-artifacts-write-one job (scanner universe board + regime
# substrate in, signals.json out), no LLM/LangGraph, pure quant transform.
# Memory 1024MB matches the main runner's headroom for the pandas/boto3
# working set the board read pulls in.
deploy_signals_envelope() {
  _deploy_image_shared_lambda "$FUNCTION_SIGNALS_ENVELOPE" "signals_envelope_handler" 300 1024
}

# OpenRouter shadow-judge scheduled runner — alpha-engine-config#2934.
# Shared image with the main runner; CMD override sets the entry to
# openrouter_shadow_handler.handler. Thin wrapper around
# evals.openrouter_shadow.run_shadow_judge_over_date (crucible-research#470)
# — invoked ONLY by EventBridge (infrastructure/setup-openrouter-shadow-
# schedule.sh), never by the production Batches-API SF chain (deliberately
# out of scope, see config#2934 / config#2575's deferred-items list).
# Timeout 900s (Lambda max) is generous headroom for a week's capture
# partition of sequential OpenRouter judge calls — mirrors thinktank's
# timeout choice for the same "EventBridge->Lambda first" reasoning.
# Memory 1024MB matches the main runner's boto3/pandas working set.
# OPENROUTER_API_KEY resolves at import time via config.py's existing
# SSM-first get_secret() chokepoint — no function-level env var config
# or bespoke secrets hydration needed (same as eval_judge_handler.py).
deploy_openrouter_shadow() {
  _deploy_image_shared_lambda "$FUNCTION_OPENROUTER_SHADOW" "openrouter_shadow_handler" 900 1024
}

# Weekly judge-sensitivity scorecard Lambda — config#752 Phase B (Brian's
# 2026-07-22 operator ruling: provision this + the SF wiring). Shared image
# with the main runner — specifically the eval-judge CMD family, because the
# synthetic-perturbation battery makes live Anthropic calls (the no-LLM
# eval-rolling-mean Lambda can't host it); CMD override sets the entry to
# perturbation_battery_handler.handler. Timeout 900s (Lambda max) + memory
# 1024MB match deploy_eval_judge — the closest analog workload (reference +
# N corrupted variants, each judged). Invoked by the Saturday SF's
# PerturbationBattery state (nousergon-data infrastructure/step_function.json),
# sequenced alongside the other eval-judge-chain observability stages.
deploy_perturbation_battery() {
  _deploy_image_shared_lambda "$FUNCTION_PERTURBATION_BATTERY" "perturbation_battery_handler" 900 1024
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

case "$TARGET" in
  main)                  build_and_deploy_main ;;
  alerts)                build_and_deploy_alerts ;;
  eval_judge)            deploy_eval_judge ;;
  eval_judge_batch)      deploy_eval_judge_batch ;;
  eval_rolling_mean)     deploy_eval_rolling_mean ;;
  rationale_clustering)  deploy_rationale_clustering ;;
  aggregate_costs)       deploy_aggregate_costs ;;
  scanner)               deploy_scanner ;;
  signals_envelope)      deploy_signals_envelope ;;
  openrouter_shadow)     deploy_openrouter_shadow ;;
  perturbation_battery)  deploy_perturbation_battery ;;
  both)                  build_and_deploy_main; build_and_deploy_alerts ;;  # ci-deploy-guard: manual — aggregate convenience target
  all)                   build_and_deploy_main; build_and_deploy_alerts; deploy_eval_judge; deploy_eval_judge_batch; deploy_eval_rolling_mean; deploy_rationale_clustering; deploy_aggregate_costs; deploy_scanner; deploy_signals_envelope; deploy_openrouter_shadow; deploy_perturbation_battery ;;  # ci-deploy-guard: manual — aggregate convenience target
  *)                     echo "Usage: $0 [main|alerts|eval_judge|eval_judge_batch|eval_rolling_mean|rationale_clustering|aggregate_costs|scanner|signals_envelope|openrouter_shadow|perturbation_battery|both|all]"; exit 1 ;;
esac

echo ""
echo "Deployment complete."
echo ""
