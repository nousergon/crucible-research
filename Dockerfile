FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Install git — required for ``pip install git+https://...`` of
# alpha-engine-lib below. The Lambda Python 3.12 base image does not
# include git; pip's git-cloning command fails with "Cannot find
# command 'git'" without this. Caught 2026-05-03 when first deploy
# after the lib-public flip (PR #103/#105) failed at the lib-install
# step. ``microdnf`` is the AL2023 minimal package manager; ``-y``
# auto-confirms. Image-size impact: ~25MB for git + git-core deps.
RUN microdnf install -y git && microdnf clean all

# Install dependencies. alpha-engine-lib is installed from public git+https
# (lib was flipped public 2026-05-03; previous versions vendored a local
# copy via deploy.sh staging). [arcticdb] pulls arcticdb (used by data/
# fetchers/price_fetcher.py); [flow_doctor] pulls flow-doctor for the
# handler's setup_logging call; [rag] pulls psycopg2-binary + pgvector +
# numpy for the qual analyst's `query_filings` tool which calls
# `alpha_engine_lib.rag.retrieve()`. Excludes pytest / python-dotenv /
# pre-installed Lambda runtime deps (boto3 etc.).
#
# IMPORTANT: keep this `@vX.Y.Z` tag in sync with the pin in
# `requirements.txt` — the `grep -vE "...|^alpha-engine-lib"` line below
# strips the lib pin from requirements before the `pip install -r`, so
# this hardcoded line is the AUTHORITATIVE pin for the Lambda image. A
# requirements-only bump won't propagate. Surfaced 2026-05-06 when a
# `@v0.4.0 → @v0.5.1` requirements bump landed but the image kept
# installing v0.3.0 (no `agent_schemas` module → ModuleNotFoundError on
# Research Lambda invocation). Treat `Dockerfile` + `Dockerfile.alerts`
# + `requirements.txt` as one tri-state pin that must move in lockstep.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir "alpha-engine-lib[arcticdb,flow_doctor,rag] @ git+https://github.com/cipher813/alpha-engine-lib@v0.32.0" && \
    grep -vE "^#|^$|^pytest|^python-dotenv|^boto3|^botocore|^s3transfer|^alpha-engine-lib" requirements.txt > /tmp/req-lambda.txt && \
    pip install --no-cache-dir -r /tmp/req-lambda.txt && \
    rm -rf /root/.cache/pip /tmp/req-lambda.txt

# Copy application code
COPY agents/ ${LAMBDA_TASK_ROOT}/agents/
COPY config/ ${LAMBDA_TASK_ROOT}/config/
COPY config.py ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/
COPY emailer/ ${LAMBDA_TASK_ROOT}/emailer/
COPY graph/ ${LAMBDA_TASK_ROOT}/graph/
COPY scoring/ ${LAMBDA_TASK_ROOT}/scoring/
COPY thesis/ ${LAMBDA_TASK_ROOT}/thesis/
COPY archive/ ${LAMBDA_TASK_ROOT}/archive/
COPY evals/ ${LAMBDA_TASK_ROOT}/evals/
COPY memory/ ${LAMBDA_TASK_ROOT}/memory/
COPY rag/ ${LAMBDA_TASK_ROOT}/rag/
# scripts/ holds aggregate_costs.py — imported by lambda/handler.py at the
# end of every successful run to write the daily cost parquet (PR #81 SF-
# wire-up). Without this COPY the import raises ModuleNotFoundError at
# runtime; the handler's try/except catches it (non-fatal — Backtester
# renders an empty cost section), but the parquet never gets written.
COPY scripts/ ${LAMBDA_TASK_ROOT}/scripts/
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/
COPY preflight.py ${LAMBDA_TASK_ROOT}/
COPY retry.py ${LAMBDA_TASK_ROOT}/
COPY health_status.py ${LAMBDA_TASK_ROOT}/
COPY dry_run.py ${LAMBDA_TASK_ROOT}/
COPY strict_mode.py ${LAMBDA_TASK_ROOT}/

# Main Lambda handler
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

# Eval-judge Lambda handlers — same image, separate Lambda functions
# in AWS that override CMD via --image-config at deploy time. Sharing
# the image avoids a parallel ECR repo + duplicate Docker build for
# handlers that need the exact same dependency set.
#
# Legacy single-Lambda handler. Retained for ad-hoc invocations,
# ``dry_run`` smoke, and the ``judge_only`` test track. The Saturday
# SF runs the batch chain (submit/poll/process) below.
COPY lambda/eval_judge_handler.py ${LAMBDA_TASK_ROOT}/eval_judge_handler.py

# Eval-judge Anthropic Message Batches API chain (ROADMAP §1642
# closure 2026-05-07). Three Lambdas share this image, each with a
# CMD override:
#   * eval_judge_submit_handler.handler   — builds + submits batch
#   * eval_judge_poll_handler.handler     — retrieves processing_status
#   * eval_judge_process_handler.handler  — streams + persists results
COPY lambda/eval_judge_submit_handler.py ${LAMBDA_TASK_ROOT}/eval_judge_submit_handler.py
COPY lambda/eval_judge_poll_handler.py ${LAMBDA_TASK_ROOT}/eval_judge_poll_handler.py
COPY lambda/eval_judge_process_handler.py ${LAMBDA_TASK_ROOT}/eval_judge_process_handler.py

# Rolling-4-week-mean Lambda handler (PR 4b) — same image, separate
# Lambda overriding CMD to ["eval_rolling_mean_handler.handler"].
COPY lambda/eval_rolling_mean_handler.py ${LAMBDA_TASK_ROOT}/eval_rolling_mean_handler.py

# Cross-week rationale clustering Lambda — same image, separate Lambda
# overriding CMD to ["rationale_clustering_handler.handler"]. Reads
# decision_artifacts/ for trailing 8 weeks, clusters rationales per
# agent_id, emits agent_rationale_template_concentration CW metric.
COPY lambda/rationale_clustering_handler.py ${LAMBDA_TASK_ROOT}/rationale_clustering_handler.py

# Daily cost aggregation Lambda — same image, CMD override to
# ["aggregate_costs_handler.handler"]. Reads decision_artifacts/_cost_raw/
# JSONL partitions for the target date, writes the daily parquet at
# decision_artifacts/_cost/{date}/cost.parquet, emits per-agent CW metrics.
# Per ROADMAP L1146 — closes the manual-trigger surface for the cost
# aggregator that PR #74 shipped (since 2026-05-01).
COPY lambda/aggregate_costs_handler.py ${LAMBDA_TASK_ROOT}/aggregate_costs_handler.py

CMD ["handler.handler"]
