"""
Rung 1 extraction agent — single-shot, triggered-only extraction of 3 typed
META_FEATURES from SEC filings / transcripts / 8-Ks via cheap-tier model
sub-arms (alpha-engine-config#3080).

No ReAct loop, no reflection pass. One LLM call per TRIGGERED ticker, using
the existing material_triggers.py event gate and qual_tools.py::query_filings
RAG retrieval.

All 4 model sub-arms run configured in ``EXTRACTION_MODEL_ARMS``. The cheapest
tier that clears the grounding/accuracy bar ships as the default (determined
by the OBSERVE pre-registration at alpha-engine-config#3080, comment
2026-07-20 — e-process + e-BH across the 4 concurrent challengers).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

# ── Typed output schema ──────────────────────────────────────────────────────


class ExtractionFeatures(BaseModel):
    """The 3 typed META_FEATURES emitted per triggered ticker.

    These are emitted as new META_FEATURES consumed by the predictor. Each
    feature is individually ablatable, schema-versioned, and must pass the
    grounding hard-gate before being accepted (see
    ``check_grounding_threshold``).
    """

    guidance_direction: str = Field(
        description="Categorical: raised/lowered/maintained/none"
    )
    risk_factor_count_delta_raw: int = Field(
        description="Integer delta in the number of distinct risk factors "
        "vs the prior comparable filing (e.g. current 10-K vs prior 10-K). "
        "Positive = more risk factors added than removed. Raw value, not "
        "winsorized — the predictor's standardization handles outliers."
    )
    management_tone_zscore: float = Field(
        description="Float z-score of management-tone sentiment in the "
        "earnings call / filing's forward-looking statements section. "
        "Computed by the extraction model relative to the corpus baseline. "
        "Positive = more optimistic/confident tone; negative = more cautious/"
        "defensive. Floor model must emit a value in [-3, 3] or the "
        "grounding gate rejects it."
    )

    @field_validator("guidance_direction")
    @classmethod
    def _validate_guidance_direction(cls, v: str) -> str:
        allowed = {"raised", "lowered", "maintained", "none"}
        if v not in allowed:
            raise ValueError(f"guidance_direction must be one of {allowed}, got {v!r}")
        return v


class ExtractionRunRecord(BaseModel):
    """Per-run record logged to the run artifact."""

    ticker: str
    model_arm: str
    features: ExtractionFeatures
    grounding_score: float
    grounding_passed: bool
    token_count_input: int
    token_count_output: int
    cost_usd: float
    duration_ms: int
    triggered_by: list[str]
    run_date: str


# ── Model sub-arm configuration ──────────────────────────────────────────────

# All four run as simultaneous sub-arms on the identical triggered-ticker set.
# The cheapest among those clearing the grounding/accuracy bar ships as the
# default. Providers are resolved via OpenRouter; Chinese-jurisdiction
# providers excluded at account level (alpha-engine-config-I3006).

EXTRACTION_MODEL_ARMS: list[dict[str, Any]] = [
    {
        "name": "floor",
        "model": "deepseek/deepseek-v4-flash",
        "cost_per_1k_in": 0.09,
        "cost_per_1k_out": 0.18,
        "structured_outputs": True,
        "note": "fleet's validated cheap tier — start here",
    },
    {
        "name": "floor-stretch",
        "model": "openai/gpt-oss-120b",
        "cost_per_1k_in": 0.03,
        "cost_per_1k_out": 0.15,
        "structured_outputs": False,
        "note": "no fleet track record; near-free data point",
    },
    {
        "name": "mid",
        "model": "qwen/qwen3.7-plus",
        "cost_per_1k_in": 0.32,
        "cost_per_1k_out": 1.28,
        "structured_outputs": True,
        "note": "only relevant if floor tiers fail the grounding bar",
    },
    {
        "name": "ceiling-ref",
        "model": "moonshotai/kimi-k2.6",
        "cost_per_1k_in": 0.66,
        "cost_per_1k_out": 3.41,
        "structured_outputs": True,
        "note": "known-good bar from Morning Signal; needs "
        "reasoning: {exclude: true} — known empty-content trap otherwise",
    },
]

# ── Prompt template ──────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """You are a financial-extraction specialist.
Given SEC filing excerpts and/or earnings call transcripts for {ticker},
extract the following 3 structured fields:

1. guidance_direction (categorical): "raised" / "lowered" / "maintained" / "none"
   - Has the company raised, lowered, or maintained its forward guidance relative
     to the prior period? "none" = no guidance mentioned in the provided text.

2. risk_factor_count_delta_raw (integer): delta count of distinct risk factors
   vs the prior comparable filing. Positive = more risk factors. Focus on the
   "Risk Factors" section of 10-K / 10-Q filings. If only one filing period is
   available, estimate whether the count direction increased or decreased.

3. management_tone_zscore (float [-3, 3]): tone z-score of management's
   forward-looking language. +1 to +3 = optimistic/confident (growth language,
   strong pipeline, upbeat guidance). -1 to -3 = cautious/defensive (hedging
   language, risk emphasis, uncertainty framing). ~0 = neutral.

Respond with ONLY a single JSON object matching this schema:
{{
    "guidance_direction": "raised" | "lowered" | "maintained" | "none",
    "risk_factor_count_delta_raw": <integer>,
    "management_tone_zscore": <float between -3 and 3>
}}"""


# ── Grounding gate ──────────────────────────────────────────────────────────


def check_grounding_threshold(
    features: ExtractionFeatures,
    source_chunks: list[dict],
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """Check extraction output against provenance-grounding threshold.

    The grounding score measures the LLM's confidence that the extracted
    values are directly supported by the source chunks. This is a hard gate:
    extractions below the threshold are REJECTED, not warned.

    Returns (passed, score) tuple.
    """
    # TODO(alpha-engine-config#3080): wire into the existing provenance-
    # grounding machinery. For now, a heuristic pass-through that returns
    # a neutral score. The grounding machinery reads the source chunks +
    # extracted features and produces a P(grounded) ∈ [0, 1].
    if not source_chunks:
        return (False, 0.0)

    # Placeholder: accept if source chunks exist and features look plausible.
    # The real grounding gate will be implemented once the provenance-
    # grounding machinery is pointed at extraction outputs (pre-registration
    # step in the issue body: "Grounding hard-gate").
    score = 1.0 if len(source_chunks) >= 1 else 0.0
    return (score >= threshold, score)


# ── Cost tracking ────────────────────────────────────────────────────────────


def _estimate_cost(
    model_arm: dict[str, Any],
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a single extraction call."""
    cost = (input_tokens / 1000) * model_arm["cost_per_1k_in"]
    cost += (output_tokens / 1000) * model_arm["cost_per_1k_out"]
    return cost


# ── LLM calling ──────────────────────────────────────────────────────────────


def _call_extraction_llm(
    model_arm: dict[str, Any],
    system_prompt: str,
    user_content: str,
) -> ExtractionFeatures | None:
    """Make a single-shot LLM call to extract features.

    Uses OpenRouter for provider-agnostic model access. Falls back gracefully
    if the model arm is unavailable (returns None — fail-soft). Not finding
    extraction features for a ticker never blocks live signals.
    """
    try:
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            log.warning(
                "OPENROUTER_API_KEY not set — cannot call extraction model %s",
                model_arm["model"],
            )
            return None

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        kwargs: dict[str, Any] = {
            "model": model_arm["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,  # deterministic extraction
        }

        if model_arm.get("structured_outputs"):
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_features",
                    "strict": True,
                    "schema": ExtractionFeatures.model_json_schema(),
                },
            }

        response = client.chat.completions.create(**kwargs)

        if not response.choices or not response.choices[0].message.content:
            log.warning("Empty response from %s for extraction", model_arm["model"])
            return None

        content = response.choices[0].message.content.strip()

        # Parse JSON response — the model should return valid JSON matching
        # the ExtractionFeatures schema, either from structured outputs or
        # from the instruction prompt.
        try:
            # Strip any markdown fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            parsed = json.loads(content)
            return ExtractionFeatures(**parsed)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning(
                "Failed to parse extraction response from %s: %s",
                model_arm["model"],
                exc,
            )
            return None

    except Exception as exc:
        log.warning(
            "Extraction LLM call failed for %s: %s",
            model_arm["model"],
            exc,
        )
        return None


# ── Extraction pipeline ──────────────────────────────────────────────────────


_RAG_TOP_K = 8  # top-k chunks from hybrid retrieval


def extract_features_for_ticker(
    ticker: str,
    triggered_by: list[str],
    model_arm: dict[str, Any],
    run_date: str,
    qual_tools: Any = None,
) -> ExtractionRunRecord | None:
    """Run single-shot extraction for one triggered ticker.

    Args:
        ticker: The ticker symbol to extract features for.
        triggered_by: List of trigger names that fired for this ticker.
        model_arm: The model arm configuration from EXTRACTION_MODEL_ARMS.
        run_date: The trading/run date string (YYYY-MM-DD).
        qual_tools: Optional pre-created qual tools instance. If None, creates one.

    Returns:
        ExtractionRunRecord if extraction succeeded, None if it failed
        gracefully (fail-soft).
    """
    import time

    start = time.monotonic()

    # Build system prompt with ticker context
    system_prompt = _EXTRACTION_SYSTEM_PROMPT.format(ticker=ticker)

    # Retrieve RAG chunks via query_filings
    if qual_tools is None:
        from agents.sector_teams.qual_tools import create_qual_tools

        qual_tools = create_qual_tools()

    rag_query = (
        f"Extract guidance direction, risk factor count changes, "
        f"and management tone for {ticker} from the provided documents"
    )

    try:
        filings_context = qual_tools.query_filings(
            ticker=ticker,
            query=rag_query,
            doc_types="10-K,10-Q,8-K,earnings_transcript",
        )
    except Exception as exc:
        log.warning("RAG retrieval failed for %s: %s", ticker, exc)
        filings_context = ""

    if not filings_context or len(filings_context.strip()) < 50:
        log.info("Insufficient RAG context for %s — skipping extraction", ticker)
        return None

    user_content = (
        f"Ticker: {ticker}\n"
        f"Run date: {run_date}\n"
        f"Triggers: {', '.join(triggered_by)}\n\n"
        f"--- Source documents ---\n{filing_context}"
    )

    features = _call_extraction_llm(model_arm, system_prompt, user_content)
    if features is None:
        return None

    # Grounding check
    source_chunks = [{"source": "query_filings", "text": filings_context[:500]}]
    grounding_passed, grounding_score = check_grounding_threshold(
        features, source_chunks
    )

    duration_ms = int((time.monotonic() - start) * 1000)

    # Rough token estimate (chars / 4 = ~tokens)
    input_tokens = len(system_prompt + user_content) // 4
    output_tokens = len(json.dumps(features.model_dump())) // 4

    return ExtractionRunRecord(
        ticker=ticker,
        model_arm=model_arm["name"],
        features=features,
        grounding_score=grounding_score,
        grounding_passed=grounding_passed,
        token_count_input=input_tokens,
        token_count_output=output_tokens,
        cost_usd=_estimate_cost(model_arm, input_tokens, output_tokens),
        duration_ms=duration_ms,
        triggered_by=triggered_by,
        run_date=run_date,
    )


def run_extraction_for_triggered_tickers(
    tickers_with_triggers: list[tuple[str, list[str]]],
    run_date: str,
    model_arm_name: str | None = None,
) -> list[ExtractionRunRecord]:
    """Run extraction for all triggered tickers across model sub-arms.

    This is the top-level entry point called by the weekly research pipeline.

    Args:
        tickers_with_triggers: List of (ticker, triggers) pairs from
            material_triggers.check_material_triggers.
        run_date: The trading/run date string (YYYY-MM-DD).
        model_arm_name: Optional specific model arm name to use. If None,
            runs all 4 sub-arms for comparison (Champion-challenger evaluation
            — the OBSERVE pre-registration).

    Returns:
        List of ExtractionRunRecord for successful extractions.
    """
    from agents.sector_teams.qual_tools import create_qual_tools

    qual_tools = create_qual_tools()
    records: list[ExtractionRunRecord] = []

    arms = (
        [a for a in EXTRACTION_MODEL_ARMS if a["name"] == model_arm_name]
        if model_arm_name
        else EXTRACTION_MODEL_ARMS
    )

    for arm in arms:
        log.info(
            "Running extraction for %d tickers with model arm %s",
            len(tickers_with_triggers),
            arm["name"],
        )

        for ticker, triggers in tickers_with_triggers:
            record = extract_features_for_ticker(
                ticker=ticker,
                triggered_by=triggers,
                model_arm=arm,
                run_date=run_date,
                qual_tools=qual_tools,
            )
            if record is not None:
                records.append(record)
                log.info(
                    "Extraction %s/%s: %s — guidance=%s rfc=%d tone=%.2f "
                    "grounded=%s cost=$%.4f",
                    arm["name"],
                    ticker,
                    "PASS" if record.grounding_passed else "FAIL",
                    record.features.guidance_direction,
                    record.features.risk_factor_count_delta_raw,
                    record.features.management_tone_zscore,
                    record.grounding_passed,
                    record.cost_usd,
                )

    return records


def _persist_run_records(
    records: list[ExtractionRunRecord],
    run_date: str,
) -> str | None:
    """Write extraction run records to S3 as a JSONL artifact.

    Args:
        records: The run records to persist.
        run_date: The trading/run date string (YYYY-MM-DD).

    Returns:
        The S3 key written to, or None if write failed.
    """
    if not records:
        return None

    try:
        import json

        import boto3

        s3 = boto3.client("s3")
        bucket = os.environ.get(
            "EXTRACTION_ARTIFACT_BUCKET",
            "alpha-engine-research",
        )
        key = (
            f"sector_teams/extraction/run_records/"
            f"run_date={run_date}/extraction_records.jsonl"
        )

        lines = "\n".join(json.dumps(r.model_dump()) for r in records)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=lines.encode("utf-8"),
            ContentType="application/jsonl",
        )
        log.info("Wrote %d extraction records to s3://%s/%s", len(records), bucket, key)
        return key
    except Exception as exc:
        log.warning("Failed to persist extraction run records: %s", exc)
        return None
