"""
Saturday-replay canary — research-repo probes for alpha-engine-config#2246.
(Live end-to-end per-PR gate verification, 2026-07-15 — no functional change.)

Every Saturday weekly-SF failure of 2026-07-11 lived in a code path the
Friday dry-preflight structurally cannot reach (``dry_run_llm=true`` only
validates boot/wiring via installed stubs — see ``lambda/handler.py``
around the ``dry_run_llm`` branch). This module exercises the real
held-thesis-update and qual-analyst extraction paths against the live
Anthropic API and the live research archive, plus a deliberately-injected
validation-retry probe, so a regression in any of the three is caught
before Saturday instead of during it.

Runs from ``alpha-engine-config/infrastructure/canary_replay_spot_bootstrap.sh``
alongside the sibling data-repo probe
(``alpha-engine-data/rag/pipelines/filing_change_detection.py --key-prefix``).

Read-only by construction: only ``ArchiveManager.load_population()`` /
``load_latest_theses()`` are called (never ``upload_db()`` or any
``save_*`` method), and every LLM call uses ``team_id="canary"`` with a
fixed synthetic ``run_date`` far outside any real archive window — so a
canary run cannot corrupt or shadow production research state even though
it calls the real production functions with real data.

CLI: python -m agents.canary_replay --run-id RUN_ID [--n-tickers 5] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from typing import Literal

from pydantic import BaseModel

log = logging.getLogger(__name__)

# Fixed sentinel run_date — a real NYSE trading day far enough in the past
# that no live archive/thesis/population history references it. Chosen
# over a moving "today + 1" date because a moving date (a) risks landing
# on a real trading day and colliding with next week's real archive keys,
# and (b) isn't reproducible run-over-run, which the drill acceptance
# test (issue #2246 closes-when) depends on. Verified at call time below
# rather than trusted blindly, in case trading-calendar data ever changes.
CANARY_RUN_DATE = "2019-01-04"


def _assert_sentinel_is_trading_day() -> None:
    from krepis.trading_calendar import is_trading_day

    d = date.fromisoformat(CANARY_RUN_DATE)
    if not is_trading_day(d):
        raise RuntimeError(
            f"canary_replay.CANARY_RUN_DATE={CANARY_RUN_DATE!r} is not a "
            "real NYSE trading day per krepis.trading_calendar.is_trading_day "
            "— update the sentinel before this canary can run."
        )


def _probe_result(name: str, status: str, detail: str, duration_s: float) -> dict:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "duration_s": round(duration_s, 2),
    }


def _load_held_tickers(am, n: int) -> list[dict]:
    """Top-N held tickers by ``long_term_score`` — the same source
    production uses (``graph/research_graph.py`` fetch_data node calls
    ``am.load_population()`` and takes every ticker; the canary narrows to
    the top N to bound LLM cost)."""
    return am.load_population()[:n]


def probe_thesis_update(am, tickers: list[dict]) -> dict:
    """Probe 2a: held-thesis update for real held tickers, live LLM.

    Calls the real ``_update_thesis_for_held_stock`` (which itself routes
    through ``invoke_structured_with_validation_retry`` internally) —
    exercises prompt loading, RAG-context wiring, and structured-output
    extraction end to end.
    """
    from agents.sector_teams.sector_team import _update_thesis_for_held_stock

    start = time.monotonic()
    try:
        symbols = [p["ticker"] for p in tickers]
        prior_theses = am.load_latest_theses(symbols)
        updated = []
        for p in tickers:
            ticker = p["ticker"]
            result = _update_thesis_for_held_stock(
                ticker=ticker,
                triggers=["canary probe — synthetic replay trigger"],
                prior_thesis=prior_theses.get(ticker),
                news_data=None,
                analyst_data=None,
                run_date=CANARY_RUN_DATE,
                team_id="canary",
            )
            updated.append({"ticker": ticker, "rating": result.get("rating")})
        return _probe_result(
            "thesis_update",
            "PASS",
            f"updated {len(updated)}/{len(tickers)} held tickers: {updated}",
            time.monotonic() - start,
        )
    except Exception as e:
        log.exception("[canary_replay] thesis_update probe failed")
        return _probe_result(
            "thesis_update", "FAIL", f"{type(e).__name__}: {e}", time.monotonic() - start
        )


def probe_qual_analyst(am, tickers: list[dict]) -> dict:
    """Probe 2b: qual-analyst ReAct extraction for the same held tickers,
    live LLM + live tool-use loop (news/filings/insider/RAG tools)."""
    from agents.sector_teams.qual_analyst import run_qual_analyst

    start = time.monotonic()
    try:
        symbols = [p["ticker"] for p in tickers]
        prior_theses = am.load_latest_theses(symbols)
        quant_top5 = [
            {
                "ticker": p["ticker"],
                "score": p.get("long_term_score", 0),
                "sector": p.get("sector"),
            }
            for p in tickers
        ]
        result = run_qual_analyst(
            team_id="canary",
            quant_top5=quant_top5,
            prior_theses=prior_theses,
            market_regime="neutral",
            run_date=CANARY_RUN_DATE,
        )
        n = len(result.get("assessments", []))
        return _probe_result(
            "qual_analyst", "PASS", f"{n} assessments returned", time.monotonic() - start
        )
    except Exception as e:
        log.exception("[canary_replay] qual_analyst probe failed")
        return _probe_result(
            "qual_analyst", "FAIL", f"{type(e).__name__}: {e}", time.monotonic() - start
        )


class _CanaryConfidenceProbe(BaseModel):
    """Deliberately tight schema mirroring the 2026-05-24 incident class
    that motivated ``invoke_structured_with_validation_retry`` in the
    first place (see ``agents/langchain_utils.py`` module docstring): a
    ``Literal`` enum paired with a prompt whose semantically-correct
    answer is NOT one of the enum values, reliably forcing at least one
    validation failure so the retry/recovery path is actually exercised
    against the live API — not just covered by
    ``tests/test_invoke_structured_with_validation_retry.py``'s mocked
    happy/recovery/terminal-failure paths.
    """

    confidence: Literal["low", "medium", "high"]
    reasoning: str


def probe_validation_retry(api_key: str | None) -> dict:
    """Probe 3: deliberately-injected validation failure through the
    shared ``invoke_structured_with_validation_retry`` chokepoint (issue
    #2246's third probe) — confirms the retry/recovery path recovers
    weekly, rather than relying on a real thesis-update call happening to
    trip it (which it may or may not do on any given run)."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    from agents.langchain_utils import (
        SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
        invoke_structured_with_validation_retry,
    )
    from agents.prompt_loader import load_prompt
    from config import ANTHROPIC_API_KEY, MAX_TOKENS_STRATEGIC, PER_STOCK_MODEL

    start = time.monotonic()
    try:
        llm = ChatAnthropic(
            model=PER_STOCK_MODEL,
            anthropic_api_key=api_key or ANTHROPIC_API_KEY,
            max_tokens=MAX_TOKENS_STRATEGIC,
            default_request_timeout=SECTOR_TEAM_LLM_REQUEST_TIMEOUT_SECONDS,
        )
        structured_llm = llm.with_structured_output(
            _CanaryConfidenceProbe, include_raw=True
        )
        # Prompt text lives in alpha-engine-config (research/prompts/
        # canary_validation_retry_probe.txt) — same load_prompt() chokepoint
        # every other prompt in this repo uses, and keeps this file itself
        # prompt-free (per this repo's CLAUDE.md: any .py file embedding an
        # LLM prompt template must be gitignored; this module deliberately
        # stays tracked/public).
        prompt = load_prompt("canary_validation_retry_probe").text
        resp = invoke_structured_with_validation_retry(
            structured_llm,
            [HumanMessage(content=prompt)],
            label="canary_replay:validation_retry",
        )
        parsed = resp.get("parsed")
        parsing_error = resp.get("parsing_error")
        if parsing_error is not None or parsed is None:
            return _probe_result(
                "validation_retry",
                "FAIL",
                f"terminal validation failure after retries: {parsing_error}",
                time.monotonic() - start,
            )
        return _probe_result(
            "validation_retry",
            "PASS",
            f"resolved to confidence={parsed.confidence!r}",
            time.monotonic() - start,
        )
    except Exception as e:
        log.exception("[canary_replay] validation_retry probe failed")
        return _probe_result(
            "validation_retry", "FAIL", f"{type(e).__name__}: {e}", time.monotonic() - start
        )


def run_canary(run_id: str, n_tickers: int = 5, api_key: str | None = None) -> dict:
    _assert_sentinel_is_trading_day()

    from archive.manager import ArchiveManager

    started_at = time.time()
    am = ArchiveManager()
    am.download_db()
    tickers = _load_held_tickers(am, n_tickers)
    if not tickers:
        # An empty population is itself a signal worth failing loud on —
        # degrading to synthetic tickers would silently stop testing the
        # real held-stock path the moment production population is empty.
        raise RuntimeError(
            "canary_replay: research.db population is empty — cannot probe "
            "held-ticker paths against real data."
        )

    probes = [
        probe_thesis_update(am, tickers),
        probe_qual_analyst(am, tickers),
        probe_validation_retry(api_key),
    ]
    overall = "PASS" if all(p["status"] == "PASS" for p in probes) else "FAIL"
    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": time.time(),
        "synthetic_run_date": CANARY_RUN_DATE,
        "held_tickers_probed": [p["ticker"] for p in tickers],
        "probes": probes,
        "overall_status": overall,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Saturday-replay canary — research-repo probes (alpha-engine-config#2246)"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-tickers", type=int, default=5)
    parser.add_argument(
        "--out", type=str, default=None, help="local path to write the result JSON"
    )
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    result = run_canary(args.run_id, n_tickers=args.n_tickers, api_key=args.api_key)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Grep-able by the orchestrating shell script (mirrors the
    # RESULT_JSON= convention added to filing_change_detection.py).
    print(f"RESULT_JSON={json.dumps(result, default=str)}")

    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
