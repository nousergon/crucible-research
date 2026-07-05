FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Install git — required for ``pip install git+https://...`` of
# alpha-engine-lib below. The Lambda Python 3.12 base image does not
# include git; pip's git-cloning command fails with "Cannot find
# command 'git'" without this. Caught 2026-05-03 when first deploy
# after the lib-public flip (PR #103/#105) failed at the lib-install
# step. ``microdnf`` is the AL2023 minimal package manager; ``-y``
# auto-confirms. Image-size impact: ~25MB for git + git-core deps.
RUN microdnf install -y git && microdnf clean all

# Bake the source commit SHA into the image so the decision-capture provenance
# stamp (``DecisionArtifact.code_sha``, L4567 sub-item 1b / #781) records the
# exact deployed code that produced each decision — the SOTA run=code+data
# reproducibility contract. ``graph/research_graph.py`` reads this at capture
# time via ``os.environ.get("ALPHA_ENGINE_CODE_SHA")``; without it the stamp is
# permanently ``None`` in prod. Passed by ``infrastructure/deploy.sh`` via
# ``--build-arg GIT_SHA=<sha>`` (CI uses ``$GITHUB_SHA``; local dev falls back
# to ``git rev-parse HEAD``). The build-arg default is left empty so a raw
# ``docker build`` that forgets to pass it stamps a falsy value (env-var-absent
# semantics — the capture read treats empty as unset → ``None``) rather than a
# misleading literal ``"unknown"``. Mirrors the predictor-side GIT_SHA wire-in.
ARG GIT_SHA=
ENV ALPHA_ENGINE_CODE_SHA=${GIT_SHA}

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
# `requirements.txt` — the `grep -vE "...|^nousergon-lib"` line below
# strips the lib pin from requirements before the `pip install -r`, so
# this hardcoded line is the AUTHORITATIVE pin for the Lambda image. A
# requirements-only bump won't propagate. Surfaced 2026-05-06 when a
# `@v0.4.0 → @v0.5.1` requirements bump landed but the image kept
# installing v0.3.0 (no `agent_schemas` module → ModuleNotFoundError on
# Research Lambda invocation). Treat `Dockerfile` + `Dockerfile.alerts`
# + `requirements.txt` as one tri-state pin that must move in lockstep.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir "nousergon-lib[arcticdb,flow_doctor,rag,contracts] @ git+https://github.com/nousergon/nousergon-lib@v0.83.0" && \
    grep -vE "^#|^$|^pytest|^python-dotenv|^boto3|^botocore|^s3transfer|^nousergon-lib" requirements.txt > /tmp/req-lambda.txt && \
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
# producers/ holds the challenger research producers (no_agent_quant /
# single_agent_quant) the handler runs as the observe-mode shadow (config#1223
# / #1403). Omitting this COPY is exactly why signals_shadow/ was empty: the
# handler's `from producers.runner import run_challengers` raised
# ModuleNotFoundError every Saturday, swallowed by the best-effort guard.
COPY producers/ ${LAMBDA_TASK_ROOT}/producers/
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
# thinktank/ is the daily research think-tank package (config#1579). The
# thinktank Lambda shares this image with a CMD override to
# thinktank_handler.handler; its private config (research/thinktank.yaml)
# and prompts are staged into config/ by infrastructure/deploy.sh exactly
# like scoring.yaml / universe.yaml / the agent prompts.
COPY thinktank/ ${LAMBDA_TASK_ROOT}/thinktank/
COPY flow-doctor.yaml ${LAMBDA_TASK_ROOT}/
COPY preflight.py ${LAMBDA_TASK_ROOT}/
COPY retry.py ${LAMBDA_TASK_ROOT}/
COPY health_status.py ${LAMBDA_TASK_ROOT}/
COPY dry_run.py ${LAMBDA_TASK_ROOT}/
COPY strict_mode.py ${LAMBDA_TASK_ROOT}/
# observe_alerts.py is a repo-ROOT single-file module imported TRANSITIVELY
# (producers/runner.py + scoring/leaderboard_producers.py). Omitting it
# import-killed the whole challenger post-step on 2026-07-03 — the second
# instance of the #340 packaging class (config#1683). The packaging guard
# (tests/test_dockerfile_packaging.py) now walks the transitive import graph
# so any new root module/package missing a COPY fails CI.
COPY observe_alerts.py ${LAMBDA_TASK_ROOT}/
COPY ops_alerts.py ${LAMBDA_TASK_ROOT}/

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

# Standalone scanner Lambda — same image, CMD override to
# ["scanner_handler.handler"]. Quant-filters S&P 500+400 (~903 tickers)
# down to ~60 candidates and writes
# s3://alpha-engine-research/candidates/{run_date}/candidates.json.
# Per ROADMAP L1995 Phase 1 — observe-only ship; Research Lambda still
# runs its internal scanner. Phase 5 (later) cuts Research over to read
# the artifact + retires the internal scanner.
COPY lambda/scanner_handler.py ${LAMBDA_TASK_ROOT}/scanner_handler.py

# Daily think-tank Lambda — same image, CMD override to
# ["thinktank_handler.handler"]. Runs `thinktank.run.run_daily()` on the
# EventBridge daily schedule (alpha-research-thinktank-daily, 14:30 UTC,
# 7 days/week): top-5 uncovered thesis builds + events sweep + churn-gated
# theme updates. Per config#1579 P1 (the EPIC's "EventBridge→Lambda first;
# EC2-spot if a run breaches ~12 min" runner decision).
COPY lambda/thinktank_handler.py ${LAMBDA_TASK_ROOT}/thinktank_handler.py

CMD ["handler.handler"]
