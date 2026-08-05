"""
Rung 1: single-shot triggered-only extraction agent.

For each ticker with a material trigger (news volume spike, >2×ATR move,
earnings proximity, insider clustering, sector regime change), makes ONE
single-shot LLM call with RAG context from filings/transcripts and extracts
3 typed META_FEATURES consumed by the predictor layer-2 ensemble.

Four model sub-arms run on an identical triggered-ticker set; whichever
tier is cheapest among those clearing the grounding/accuracy bar ships as
the default. All via OpenRouter (Chinese-jurisdiction providers excluded
at account level per config-I3006).

This is the substrate for the agentic-evolution ladder (alpha-engine-config#3079).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Cost-attribution join key (krepis >= 0.23) ──────────────────────────────

CALLSITE_ID = "rung1-extraction"
"""Cost-attribution join key for this call site. Must match an entry in
alpha-engine-config/private-docs/LLM_CALLSITE_REGISTRY.yaml."""

# ── Model sub-arms ──────────────────────────────────────────────────────────

EXTRACTION_MODEL_ARMS: dict[str, str] = {
    "floor": "deepseek/deepseek-v4-flash",
    "floor-stretch": "openai/gpt-oss-120b",
    "mid": "qwen/qwen3.7-plus",
    "ceiling": "moonshotai/kimi-k2.6",
}
"""Four concurrent model sub-arms run in parallel on the same triggered-
ticker set. The cheapest tier clearing the grounding/accuracy bar ships
as the default (enforced by the e-BH multiplicity procedure, pre-registered
in EXPERIMENTS.md).

Pricing (approximate ~$/1M in/out):
  floor:        0.09 / 0.18  (deepseek-v4-flash — fleet's validated cheap tier)
  floor-stretch: 0.03 / 0.15 (gpt-oss-120b — near-free data point)
  mid:          0.32 / 1.28  (qwen3.7-plus — only relevant if floor tiers fail)
  ceiling:      0.66 / 3.41  (kimi-k2.6 — known-good bar from Morning Signal)
"""

DEFAULT_MODEL_ARM = "floor"

# ── Grounding gate ──────────────────────────────────────────────────────────

GROUNDING_THRESHOLD = 0.7
"""Extractions below this provenance-grounding score are REJECTED, not warned.
A hallucinated number feeds the model uncorrected (unlike a hallucinated
sentence), so the grounding hard-gate is mandatory — point the existing
provenance-grounding machinery at the extraction outputs."""

GROUNDING_ATTEMPTS = 2
"""Number of structured-output extraction attempts before raising."""


# ── Structured output schema ────────────────────────────────────────────────

class ExtractionOutput(BaseModel):
    """Schema-versioned output for one triggered ticker's single-shot LLM call.

    All three fields are consumed as META_FEATURES by the predictor layer-2
    ensemble. Units-suffix rule applies (alpha-engine-config policy) — each
    column has a SCHEMA.md §3 row + FeatureEntry in registry.py::CATALOG.
    """

    guidance_direction: Literal["raised", "lowered", "maintained", "none"] = Field(
        ...,
        description=(
            "Earnings guidance direction extracted from the most recent "
            "filing/transcript. One of: 'raised', 'lowered', 'maintained', "
            "'none' (no guidance mentioned or unclear)."
        ),
    )
    risk_factor_count_delta_raw: int = Field(
        ...,
        description=(
            "Change in the number of distinct risk factor headings vs the "
            "prior comparable filing (10-Q vs prior 10-Q; 10-K vs prior 10-K). "
            "Positive = more risk factors added; negative = consolidated or "
            "removed. Raw integer delta (not normalized)."
        ),
    )
    management_tone_zscore: float = Field(
        ...,
        description=(
            "Z-score of management's tone sentiment extracted from the "
            "MD&A / earnings call Q&A sections, computed against a sector "
            "baseline. Positive = more optimistic than sector peers; negative "
            "= more cautious/pessimistic. Range typically [-3.0, 3.0]."
        ),
    )


# ── Prompt templates ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a financial-analysis extraction engine. Your job is to read the provided SEC filing excerpts and earnings transcript segments for a single stock and extract exactly three structured data points.

Rules:
1. Read the context carefully — base your answers ONLY on the provided excerpts.
2. If a filing type mismatch makes a field nonsensical (e.g., no guidance in an 8-K), use the schema's "none" / 0 / 0.0 defaults.
3. Do NOT explain your reasoning — output ONLY valid JSON matching the schema.
4. The risk_factor_count_delta_raw is relative to the PRIOR comparable filing (10-Q→10-Q, 10-K→10-K), as inferred from the context. If uncertain, use 0.
5. The management_tone_zscore is relative to the sector baseline. If uncertain, use 0.0."""


def build_extraction_prompt(ticker: str, triggers: list[str], filings_context: str) -> str:
    """Build the user-content prompt for one single-shot extraction call.

    Args:
        ticker: Stock symbol (e.g. 'AAPL').
        triggers: Material-trigger names that fired (e.g. 'news_volume_spike').
        filings_context: RAG-retrieved excerpts from SEC filings / transcripts.

    Returns:
        Rendered user-content string for the LLM call.
    """
    trigger_lines = "\n".join(f"  - {t}" for t in (triggers or ["periodic refresh"]))

    return f"""## Ticker
{ticker}

## Material triggers (why this ticker is being analysed)
{trigger_lines}

## Filing & transcript context
{filings_context}

## Extraction task
From the context above, extract:
1. **guidance_direction** — Did management raise, lower, or maintain earnings guidance? (one of: raised, lowered, maintained, none)
2. **risk_factor_count_delta_raw** — How many MORE or FEWER risk factor headings appear vs the prior comparable filing? (integer)
3. **management_tone_zscore** — What is the z-score of management's tone vs sector baseline from the relevant excerpts? (float, typically [-3.0, 3.0])

Respond with a single JSON object containing these three fields."""


# ── Extraction function ─────────────────────────────────────────────────────


def extract_single_ticker(
    ticker: str,
    triggers: list[str],
    filings_context: str,
    model_arm: str = DEFAULT_MODEL_ARM,
    *,
    api_key: str | None = None,
    client_factory: Callable | None = None,
    callsite_id: str = CALLSITE_ID,
) -> ExtractionOutput:
    """Single-shot structured extraction for one triggered ticker.

    Args:
        ticker: Stock symbol.
        triggers: Material-trigger names that fired.
        filings_context: RAG-retrieved filing/transcript excerpts.
        model_arm: Which model sub-arm to use (key into EXTRACTION_MODEL_ARMS).
        api_key: OpenRouter API key (falls back to config.OPENROUTER_API_KEY).
        client_factory: Test seam — callable ``(spec, api_key) -> transport``.
        callsite_id: krepis cost-attribution join key.

    Returns:
        ExtractionOutput with the three typed META_FEATURES.

    Raises:
        RuntimeError: No API key configured.
        ValueError: Unknown model_arm.
        krepis error: On persistent transport or parse failure.
    """
    if model_arm not in EXTRACTION_MODEL_ARMS:
        raise ValueError(
            f"Unknown model_arm={model_arm!r}. "
            f"Choose from: {', '.join(sorted(EXTRACTION_MODEL_ARMS))}"
        )

    from krepis.llm import LLMClient
    from krepis.llm_config import ModelSpec

    from config import MAX_TOKENS_PER_STOCK, OPENROUTER_API_KEY

    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise RuntimeError(
            f"Rung-1 extraction ({ticker}) requires an OpenRouter API key: "
            "pass api_key= explicitly, or ensure config.OPENROUTER_API_KEY "
            "resolves (SSM parameter /alpha-engine/OPENROUTER_API_KEY, or "
            "the OPENROUTER_API_KEY environment variable as a fallback)."
        )

    model_id = EXTRACTION_MODEL_ARMS[model_arm]
    prompt = build_extraction_prompt(ticker, triggers, filings_context)

    spec = ModelSpec(
        provider="openrouter",
        model=model_id,
        max_tokens=MAX_TOKENS_PER_STOCK,
        # REQUIRED — structured_outputs=False matches the fleet's live-verified
        # OpenRouter pattern: strict response_format=json_schema is unreliable
        # for DeepSeek-family models (they intermittently rename required fields).
        # JSON-instruction + tolerant extraction is the production-proven path.
        structured_outputs=False,
        # REQUIRED for reasoning-capable OpenRouter models — without this, the
        # model can burn its entire output budget on chain-of-thought and return
        # empty content (config#1659 / config#2575).
        reasoning={"exclude": True},
    )
    client = LLMClient(
        spec,
        api_key=key,
        client_factory=client_factory,
        max_retries=GROUNDING_ATTEMPTS,
        callsite_id=callsite_id,
    )
    result = client.structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        user_content=prompt,
        schema=ExtractionOutput,
        schema_name="rung1_extraction_output",
        attempts=GROUNDING_ATTEMPTS,
    )
    log.info(
        "[rung1] extraction ticker=%s model_arm=%s model_id=%s "
        "guidance=%s risk_delta=%d tone_z=%.3f",
        ticker, model_arm, model_id,
        result.guidance_direction, result.risk_factor_count_delta_raw,
        result.management_tone_zscore,
    )
    return result


# ── Batch extraction ────────────────────────────────────────────────────────


def retrieve_filings_context(ticker: str, query: str = "management guidance, risk factors, forward outlook") -> str:
    """Retrieve RAG context from SEC filings and transcripts for a ticker.

    A thin wrapper around ``nousergon_lib.rag.retrieve`` — the same underlying
    retrieval the existing sector-team agents use via ``qual_tools.query_filings``.

    Args:
        ticker: Stock symbol.
        query: Search query for the hybrid retriever.

    Returns:
        Formatted context string, or an error message if retrieval fails.
    """
    try:
        from datetime import date, timedelta

        from nousergon_lib.rag import retrieve

        results = retrieve(
            query=query,
            tickers=[ticker],
            doc_types=["10-K", "10-Q", "earnings_transcript"],
            min_date=date.today() - timedelta(days=730),
            top_k=8,
            method="hybrid",
            vector_weight=0.7,
        )
        if not results:
            return f"No filing data found for {ticker}."

        fragments = []
        for r in results:
            source = getattr(r, "source", "unknown")
            fragments.append(f"[{source}] {r.text[:2000]}")
        return "\n\n".join(fragments)

    except Exception as exc:
        log.warning("[rung1] RAG retrieval failed for %s: %s", ticker, exc)
        return f"RAG retrieval unavailable for {ticker}: {exc}"


def extract_triggered_tickers(
    triggered_tickers: list[tuple[str, list[str]]],
    model_arm: str = DEFAULT_MODEL_ARM,
    *,
    api_key: str | None = None,
    rag_fn: Callable[[str, str], str] | None = None,
) -> list[dict]:
    """Run extraction for all triggered tickers.

    Args:
        triggered_tickers: List of (ticker, triggers) pairs.
        model_arm: Which model sub-arm to use.
        api_key: OpenRouter API key.
        rag_fn: Optional override for RAG retrieval (default: retrieve_filings_context).

    Returns:
        List of dicts with ticker + extracted fields, ready for the predictor.
    """
    _rag_fn = rag_fn or retrieve_filings_context
    results: list[dict] = []

    for ticker, triggers in triggered_tickers:
        try:
            context = _rag_fn(ticker)
            extracted = extract_single_ticker(
                ticker, triggers, context,
                model_arm=model_arm, api_key=api_key,
            )
            results.append({
                "ticker": ticker,
                "guidance_direction": extracted.guidance_direction,
                "risk_factor_count_delta_raw": extracted.risk_factor_count_delta_raw,
                "management_tone_zscore": extracted.management_tone_zscore,
                "model_arm": model_arm,
            })
        except Exception as exc:
            log.error("[rung1] Extraction failed for %s: %s", ticker, exc)
            # Fail-soft per the design: an extraction failure never blocks live
            # signals. But fail-LOUD via the standard alert path.
            results.append({
                "ticker": ticker,
                "guidance_direction": "none",
                "risk_factor_count_delta_raw": 0,
                "management_tone_zscore": 0.0,
                "model_arm": model_arm,
                "error": str(exc),
            })

    return results
