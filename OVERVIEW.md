# alpha-engine-research — Code Index

> Index of entry points, key files, and data contracts. Companion to [README.md](README.md). System overview lives in [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs).

## Module purpose

Multi-agent investment-research pipeline — six sector teams, a CIO, a macro economist, an LLM-as-judge layer — emitting `signals.json` for the rest of the Alpha Engine system.

## Entry points

| File | What it does |
|---|---|
| [`lambda/handler.py`](lambda/handler.py) | Production Lambda entry — Saturday SF invokes this |
| [`local/run.py`](local/run.py) | Local CLI — `--dry-run`, `--stub-llm`, ticker/sector overrides |
| [`main.py`](main.py) | Orchestration entrypoint shared between Lambda + CLI |
| [`lambda/alerts_handler.py`](lambda/alerts_handler.py) | Intraday price-alert Lambda (every 30 min during market hours) |
| [`lambda/eval_judge_handler.py`](lambda/eval_judge_handler.py) | LLM-as-judge eval Lambda |

## Where things live

| Concept | File |
|---|---|
| LangGraph orchestrator (Send fan-out, fan-in, state) | [`graph/research_graph.py`](graph/research_graph.py) |
| Typed state schemas (Pydantic) | [`graph/state_schemas.py`](graph/state_schemas.py) — research-internal state + storage types; LLM-output schemas re-exported from `alpha_engine_lib.agent_schemas` (lifted 2026-05-05, lib v0.4.0) |
| State reducers | [`graph/reducers.py`](graph/reducers.py) |
| Per-call LLM cost tracker + run-budget hard ceiling | [`graph/llm_cost_tracker.py`](graph/llm_cost_tracker.py) |
| Decision-artifact capture helpers | [`graph/decision_capture_helpers.py`](graph/decision_capture_helpers.py) |
| Sector team sub-graph (quant → qual → peer review) | [`agents/sector_teams/sector_team.py`](agents/sector_teams/sector_team.py) |
| Quant analyst tool surface | [`agents/sector_teams/quant_tools.py`](agents/sector_teams/quant_tools.py) |
| Qual analyst tool surface (incl. RAG `query_filings`) | [`agents/sector_teams/qual_tools.py`](agents/sector_teams/qual_tools.py) |
| Peer review (intra-team finalization) | [`agents/sector_teams/peer_review.py`](agents/sector_teams/peer_review.py) |
| Material-trigger logic for thesis updates | [`agents/sector_teams/material_triggers.py`](agents/sector_teams/material_triggers.py) |
| GICS-to-team mapping | [`agents/sector_teams/team_config.py`](agents/sector_teams/team_config.py) |
| CIO batch evaluation (4-dim rubric, entrant gate) | [`agents/investment_committee/ic_cio.py`](agents/investment_committee/ic_cio.py) |
| Macro economist (reflection loop) | [`agents/macro_agent.py`](agents/macro_agent.py) |
| Prompt loader (frontmatter-versioned, sha256 hash) | [`agents/prompt_loader.py`](agents/prompt_loader.py) |
| Token guard | [`agents/token_guard.py`](agents/token_guard.py) |
| Composite scoring (formula only — weights private) | [`scoring/composite.py`](scoring/composite.py) |
| LLM-as-judge rubric scoring | [`evals/judge.py`](evals/judge.py) |
| Eval rolling-mean tracker | [`evals/rolling_mean.py`](evals/rolling_mean.py) |
| LangGraph trajectory invariants | [`evals/trajectory.py`](evals/trajectory.py) |
| Archive manager (S3 + SQLite + thesis history) | [`archive/manager.py`](archive/manager.py) |
| SQLite schema | [`archive/schema.py`](archive/schema.py) |
| Health status writer | [`health_status.py`](health_status.py) |
| Dry-run / stub harness | [`dry_run.py`](dry_run.py) |
| Strict-mode toggle (typed-state hard-fail) | [`strict_mode.py`](strict_mode.py) |

Proprietary files (gitignored locally; loaded at runtime from `alpha-engine-config`):

| Concept | File |
|---|---|
| Agent prompt templates | `config/prompts/*.txt` |
| Scoring weights + sub-score formulas | `scoring/technical.py`, `scoring/performance_tracker.py` |
| Universe + threshold configuration | `config/universe.yaml`, `config/scoring.yaml` |
| Population selection logic | `data/population_selector.py` |
| Quant filter pipeline | `data/scanner.py` |

## Inputs / outputs

### Reads
| Source | Path |
|---|---|
| Universe + sector ETFs (constituents) | `s3://alpha-engine-research/market_data/weekly/{date}/constituents.json` |
| Macro context | `s3://alpha-engine-research/market_data/weekly/{date}/macro.json` |
| Alternative data per ticker | `s3://alpha-engine-research/market_data/weekly/{date}/alternative/{ticker}.json` |
| RAG corpus (qual analyst tool) | Neon pgvector — `rag.documents`, `rag.chunks` |

### Writes
| Destination | Path |
|---|---|
| Signals (rest of system reads this) | `s3://alpha-engine-research/signals/{date}/signals.json` |
| Per-ticker thesis snapshots (never overwritten) | `s3://alpha-engine-research/archive/universe/{TICKER}/` |
| Buy-candidate theses | `s3://alpha-engine-research/archive/candidates/{TICKER}/` |
| Macro environment reports | `s3://alpha-engine-research/archive/macro/` |
| Morning email payload | `s3://alpha-engine-research/consolidated/{date}/morning.md` |
| Decision-capture artifacts | `s3://alpha-engine-research/decision_artifacts/{date}/{agent_id}/` |
| Per-call LLM cost JSONLs | `s3://alpha-engine-research/decision_artifacts/_cost_raw/{date}/{run_id}/` |
| LLM-as-judge eval artifacts | `s3://alpha-engine-research/eval_artifacts/{date}/` |
| Signal history + theses + IC audit trail | `s3://alpha-engine-research/research.db` (SQLite) |

## Run modes

| Mode | Where | Command |
|---|---|---|
| Production | Lambda (Docker on ECR) | `./infrastructure/deploy.sh main` then triggered by Saturday SF |
| Stub run (no API spend) | venv | `python local/run.py --stub-llm` |
| Dry run (small population, no S3 writes) | venv | `python local/run.py --dry-run --tickers AAPL,MSFT` |
| Eval-only | Lambda | `lambda/eval_judge_handler.py` invoked separately by SF eval state |

Deploy: Docker image built locally and pushed to ECR; Lambda alias `live` updated by `infrastructure/deploy.sh`.

## Tests

`pytest tests/` covers state schemas, scoring math, LangGraph trajectory invariants (`sector_team_node` runs exactly 6 times per Send), reducers, decision-artifact capture, RAG retrieval shape, prompt-versioning regression locks (~30+ locks across PRs B/C/D), strict validation, and judge rubric replay. Test surface ≈ 640 passing.
