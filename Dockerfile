FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Stage alpha-engine-lib from a local vendor directory.
# infrastructure/deploy.sh populates vendor/alpha-engine-lib from a
# sibling checkout (local dev) or GitHub Actions checkout (CI) before
# Docker build. Mirrors the alpha-engine-predictor staging pattern —
# the private repo reaches the build context without needing a GitHub
# PAT or Docker build secret. [arcticdb] pulls arcticdb (used by
# data/fetchers/price_fetcher.py); [flow_doctor] pulls flow-doctor
# for the handler's setup_logging call.
COPY vendor/alpha-engine-lib /tmp/alpha-engine-lib

# Install dependencies. Exclude pytest / python-dotenv / pre-installed
# Lambda runtime deps (boto3 etc.). Filtered list is written to a
# requirements file and consumed by `pip install -r` so pip's own parser
# handles inline comments — `$(grep ...)` shell substitution would
# word-split a `pkg # comment` line into separate args and pip would
# reject the bare `#` as an invalid requirement.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir /tmp/alpha-engine-lib[arcticdb,flow_doctor] && \
    grep -vE "^#|^$|^pytest|^python-dotenv|^boto3|^botocore|^s3transfer|^alpha-engine-lib" requirements.txt > /tmp/req-lambda.txt && \
    pip install --no-cache-dir -r /tmp/req-lambda.txt && \
    rm -rf /root/.cache/pip /tmp/alpha-engine-lib /tmp/req-lambda.txt

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
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/
COPY preflight.py ${LAMBDA_TASK_ROOT}/
COPY retry.py ${LAMBDA_TASK_ROOT}/
COPY health_status.py ${LAMBDA_TASK_ROOT}/
COPY ssm_secrets.py ${LAMBDA_TASK_ROOT}/
COPY dry_run.py ${LAMBDA_TASK_ROOT}/

# Main Lambda handler
COPY lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

CMD ["handler.handler"]
