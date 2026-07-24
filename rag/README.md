# RAG — Semantic Retrieval for Research Agents

Hybrid retrieval (vector + Full-Text Search) over SEC filings, earnings transcripts, and thesis history. Provides the qual analyst agents with deep fundamental context beyond headlines and consensus data.

> **Retrieval-only** in this repo. The shared retrieval/db/embeddings/schema code lives in [`alpha_engine_lib.rag`](https://github.com/nousergon/nousergon-lib/tree/main/src/alpha_engine_lib/rag) (since lib v0.3.0; hybrid-retrieval API since v0.6.0). RAG **ingestion** lives in [`nousergon-data/rag/pipelines/`](https://github.com/nousergon/nousergon-data/tree/main/rag/pipelines) and runs as part of the weekly Step Function via that repo's `infrastructure/spot_data_weekly.sh`.

## Architecture

```
Ingestion (weekly)          Neon pgvector + FTS              Qual Analyst Agent
alpha-engine-data    ──→   rag.documents             ──→     @tool query_filings()
SEC + 8-K + theses   ──→   rag.chunks                ──→     hybrid retrieval
                            ├─ HNSW on embedding      ──→     top-k results +
                            └─ GIN on content_tsv             component scores
```

## Retrieval methods

The lib's `retrieve()` API supports three methods (since v0.6.0):

| Method | Strong on | Weak on |
|---|---|---|
| `vector` | Conceptual / paraphrased queries (competitive moat, strategy) | Exact-term surfaces (tickers, $ amounts, filing types) |
| `keyword` | Literal-term matches (PostgreSQL FTS via `ts_rank_cd`) | Conceptual queries lacking literal overlap |
| `hybrid` | Both — blends top_k from each side, normalizes via min-max within the candidate set, returns weighted blend |

**This repo's qual analyst calls `retrieve(method="hybrid", vector_weight=0.7)`** at `agents/sector_teams/qual_tools.py::query_filings`. Per-call component scores (`vector_score` / `keyword_score` / `combined_score`) are emitted in a structured `RAG_RETRIEVE` INFO log line for decision-artifact capture and LangSmith trace observability.

`vector_weight=0.7` is the ROADMAP-spec'd starting default. Empirical calibration may move the value once enough data accumulates — see "Calibration owed" below.

## Retrieval surface used by this repo

| Caller | Imports |
|---|---|
| `agents/sector_teams/qual_tools.py` | `from alpha_engine_lib.rag import retrieve` (qual analyst's `query_filings` tool, hybrid mode) |
| `graph/research_graph.py` | `from alpha_engine_lib.rag import is_available` (gates RAG access at graph startup) |

The lib re-exports `retrieve`, `ingest_document`, `document_exists`, `embed_texts`, `get_connection`, and `is_available`. Schema is shipped as package data (`alpha_engine_lib.rag/schema.sql`); the `0001_content_tsv.sql` migration is shipped at `alpha_engine_lib.rag/migrations/`.

## Environment Variables

| Var | Purpose |
|-----|---------|
| `RAG_DATABASE_URL` | Neon pooled connection string (read by `is_available` and `retrieve`) |
| `VOYAGE_API_KEY` | Voyage embedding API key (used by retrieval-time query embedding) |

## Cost

| Component | Monthly |
|-----------|---------|
| Neon pgvector + FTS (free tier) | $0 |
| Voyage embeddings (~903 stocks × weekly) | ~$3.60 |

## Eval harness

`evals/rag_retrieval.py` + `scripts/run_rag_retrieval_eval.py` ship a 6-condition × 3-cutoff recall@k harness. Empirical calibration of `vector_weight` is the intended use case but the harness is general-purpose for any retrieval regression scan.

```bash
# Curate evals/rag_retrieval_queries.yaml with hand-picked
# (query → expected_chunk_id) pairs (or seed via
# scripts/seed_rag_retrieval_queries.py for a starting point).

python scripts/run_rag_retrieval_eval.py
# → ~/Development/alpha-engine-docs/private/rag-retrieval-eval-{date}.md
```

## Calibration result (2026-05-08)

Hand-curated test set: **25 queries** across the 6 categories (11 additional queries skipped where neither retrieval method surfaced a relevant chunk in top-10). Eval ran each query through 6 conditions (vector / keyword / hybrid w∈{0.3, 0.5, 0.7, 0.9}) at top_k=20.

### Overall recall@10

| Method | Recall@10 |
|---|---|
| vector | **0.913** |
| keyword | 0.043 |
| hybrid w=0.3 | 0.783 |
| hybrid w=0.5 | **0.913** |
| hybrid w=0.7 | **0.913** |
| hybrid w=0.9 | **0.913** |

### Findings

- **Vector alone is sufficient** at the eval-harness level. Hybrid at w∈{0.5, 0.7, 0.9} ties vector exactly — no regression, no lift.
- **Keyword alone underperforms (0.043)** because the eval harness lacks the `tickers=[…]` pre-filter that production `query_filings` passes. Without ticker narrowing, the OR-relaxed FTS pool is dominated by token-overlap noise.
- **Hybrid w=0.3 hurts** (0.783) — too much weight on the noisy keyword side drags the blend down.
- **Per-category:** vector hits 100% on abstract_thesis, conceptual_narrative, date_range, and filing_type. The 75% on ticker_named_entity (n=4) and 50% on quantitative_line_item (n=2) are small-n; not load-bearing signal.

### Default decision

`vector_weight=0.7` stays. The eval confirms it ties vector at recall@10 — adding the keyword side at this weight costs nothing observable in the harness and may help in production when the ticker pre-filter narrows the keyword pool to the queried company before ranking. Hybrid is a safety belt, not a lift driver, against this corpus and harness.

Full report at `~/Development/alpha-engine-docs/private/rag-retrieval-eval-2026-05-08.md`.

### Why the harness underestimates keyword

`agents/sector_teams/qual_tools.py::query_filings` passes `tickers=[ticker]` to `retrieve()` in production. The eval harness's YAML schema does not yet support per-query metadata filters, so retrieval runs unfiltered against the full 21,550-chunk corpus. For ticker-anchored queries (which most qual-analyst calls are), this is the wrong measurement surface. The cleaner calibration path is **passive measurement from production** — mine the `RAG_RETRIEVE` log lines + downstream agent citations once 2-3 weeks of hybrid-mode decision artifacts accumulate. Tracked as a ROADMAP P3 follow-up.
