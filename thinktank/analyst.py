"""The analyst — thesis builds and the daily events sweep (skeleton crew).

One agent, two jobs:
- ``build_thesis``: full thesis for one name (filings, news/sentiment,
  valuation, moat, risks, catalysts) consuming the CURRENT macro/sector
  themes as context rather than re-deriving them per name.
- ``sweep``: chunked pass over all covered names against today's news
  aggregates; names flagged ``update_thesis`` get a thesis update. Sweep
  output also surfaces macro-relevant developments, which feed the theme
  keeper's churn-gated daily update.
"""

from __future__ import annotations

import json
import logging

from agents.prompt_loader import load_prompt

from thinktank import THESIS_KEY_TMPL, THESIS_LATEST_TMPL
from thinktank.capture import emit_thesis_capture
from thinktank.client import ThinktankClient
from thinktank.context import ContextBundle, filings_excerpts
from thinktank.schemas import (
    CompanyThesis,
    CompanyThesisLLM,
    SweepBatchLLM,
    TickerEventAssessment,
)
from thinktank.storage import ThinktankStore
from thinktank.themes import ThemeKeeper

logger = logging.getLogger(__name__)

THESIS_TIER = "thesis"
SWEEP_TIER = "sweep"

_ANALYST_SYSTEM = (
    "You are a buy-side equity research analyst writing an institutional "
    "investment thesis. Be concrete and evidence-based: cite the specific "
    "inputs provided (filings excerpts, news, metrics, themes). Distinguish "
    "what you know from the inputs vs. general knowledge. No boilerplate."
)
_SWEEP_SYSTEM = (
    "You are the coverage-desk analyst doing the daily events sweep. For each "
    "ticker, decide from the news-aggregate signals whether anything happened "
    "that warrants re-underwriting the investment thesis. Be conservative: "
    "routine drift is action=none; action=update_thesis is for genuinely "
    "thesis-relevant developments (guidance, M&A, regulatory, major product, "
    "severe sentiment/event spikes)."
)


def load_latest_thesis(store: ThinktankStore, ticker: str) -> CompanyThesis | None:
    raw = store.get_json(THESIS_LATEST_TMPL.format(ticker=ticker))
    return CompanyThesis.model_validate(raw) if raw is not None else None


def _write_thesis(store: ThinktankStore, thesis: CompanyThesis) -> None:
    payload = thesis.model_dump()
    store.put_json(
        THESIS_KEY_TMPL.format(ticker=thesis.ticker, version=thesis.version), payload
    )
    store.put_json(THESIS_LATEST_TMPL.format(ticker=thesis.ticker), payload)


def build_thesis(
    store: ThinktankStore,
    client: ThinktankClient,
    ctx: ContextBundle,
    themes: ThemeKeeper,
    *,
    ticker: str,
    board_row: dict | None,
    trading_day: str,
    calendar_date: str,
    update_reason: str,
    event_context: str | None = None,
) -> CompanyThesis:
    """Build (or re-underwrite) the thesis for one ticker and persist it."""
    prior = load_latest_thesis(store, ticker)
    sector = (board_row or {}).get("sector") or (prior.sector if prior else None)

    sources: list[str] = []
    filings = filings_excerpts(ticker) if ctx.rag_available else []
    if filings:
        sources.append("rag_filings")
    news = ctx.news_by_ticker.get(ticker)
    if news:
        sources.append("news_aggregates")
    if board_row:
        sources.append("universe_board")
    signals_entry = ((ctx.signals or {}).get("signals") or {}).get(ticker)
    if signals_entry:
        sources.append("weekly_signals")

    next_version = (prior.version + 1) if prior else 1
    prompt = load_prompt("thinktank_thesis")
    rendered = prompt.format(
        ticker=ticker,
        sector=sector or "unknown",
        update_reason=update_reason,
        prior_thesis=prior.thesis.model_dump_json() if prior else "(none — initial coverage)",
        event_context=event_context or "(none)",
        board_row=json.dumps(_slim_board_row(board_row), default=str),
        weekly_signal=json.dumps(signals_entry or {}, default=str),
        news_aggregate=json.dumps(news or {}, default=str),
        filings_excerpts="\n---\n".join(filings) or "(no filings context available)",
        macro_theme=themes.macro_summary(),
        sector_theme=themes.sector_summary(sector),
    )
    result = client.complete(
        THESIS_TIER,
        agent_id="analyst_thesis",
        system=_ANALYST_SYSTEM,
        user=rendered,
        response_model=CompanyThesisLLM,
        prompt_id=prompt.name,
        prompt_version=prompt.version,
        sft_meta={
            "ticker": ticker,
            "thesis_version": next_version,
            "update_reason": update_reason,
            "trading_day": trading_day,
            "capture_run_id": f"{client.run_id}-{ticker}-v{next_version}",
        },
    )

    macro_v, sector_vs = themes.theme_versions()
    thesis = CompanyThesis(
        ticker=ticker,
        version=next_version,
        trading_day=trading_day,
        calendar_date=calendar_date,
        update_reason=update_reason,  # type: ignore[arg-type]
        thesis=result.parsed,
        sector=sector,
        attractiveness_score=(board_row or {}).get("attractiveness_score"),
        attractiveness_rank=(board_row or {}).get("_attractiveness_rank"),
        macro_theme_version=macro_v,
        sector_theme_version=sector_vs.get(sector or ""),
        sources_used=sources,
        event_context=event_context,
        model=result.model,
        tier=result.tier,
        prompt_version=prompt.version,
        cost_usd=result.cost_usd,
    )
    _write_thesis(store, thesis)
    emit_thesis_capture(
        base_run_id=client.run_id,
        ticker=ticker,
        version=thesis.version,
        result=result,
        system=_ANALYST_SYSTEM,
        user=rendered,
        prompt_version_hash=prompt.hash,
        input_data_snapshot={
            "ticker": ticker,
            "update_reason": update_reason,
            "event_context": event_context,
            "board_row": _slim_board_row(board_row),
            "weekly_signal": signals_entry or {},
            "news_aggregate": news or {},
            "filings_excerpts": filings,
            "macro_theme": themes.macro_summary(),
            "sector_theme": themes.sector_summary(sector),
            "prior_thesis": prior.thesis.model_dump() if prior else None,
        },
        agent_output=thesis.model_dump(),
        bucket=store.bucket,
        s3_client=store.s3,
    )
    logger.info(
        "thesis written %s v%d (%s, stance=%s conviction=%d, $%.4f)",
        ticker,
        thesis.version,
        update_reason,
        result.parsed.stance,
        result.parsed.conviction,
        result.cost_usd,
    )
    return thesis


def _slim_board_row(row: dict | None) -> dict:
    if not row:
        return {}
    keep = (
        "ticker",
        "sector",
        "industry",
        "attractiveness_score",
        "pillars",
        "focus_score",
        "tech_score",
        "metrics",
        "tradeability",
    )
    return {k: row.get(k) for k in keep if k in row}


def sweep(
    client: ThinktankClient,
    ctx: ContextBundle,
    *,
    covered: list[str],
    chunk_size: int,
) -> tuple[list[TickerEventAssessment], str]:
    """Assess all covered names in chunks. Returns (assessments, macro_notes)."""
    prompt = load_prompt("thinktank_sweep")
    assessments: list[TickerEventAssessment] = []
    macro_notes: list[str] = []
    for i in range(0, len(covered), chunk_size):
        chunk = covered[i : i + chunk_size]
        rows = {t: ctx.news_by_ticker.get(t) or {} for t in chunk}
        rendered = prompt.format(
            tickers=", ".join(chunk),
            news_rows=json.dumps(rows, default=str),
            market_regime=ctx.market_regime(),
        )
        result = client.complete(
            SWEEP_TIER,
            agent_id="analyst_sweep",
            system=_SWEEP_SYSTEM,
            user=rendered,
            response_model=SweepBatchLLM,
            prompt_id=prompt.name,
            prompt_version=prompt.version,
            sft_meta={"tickers": chunk},
        )
        batch: SweepBatchLLM = result.parsed  # type: ignore[assignment]
        known = set(chunk)
        for a in batch.assessments:
            if a.ticker in known:
                assessments.append(a)
            else:
                logger.warning("sweep returned unknown ticker %s — dropped", a.ticker)
        if batch.macro_relevant.strip():
            macro_notes.append(batch.macro_relevant.strip())
    return assessments, "\n".join(macro_notes)
