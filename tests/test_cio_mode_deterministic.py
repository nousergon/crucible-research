"""Tests for the cio_mode=deterministic bypass (config#799).

Pins that when the backtester writes cio_mode="deterministic" to S3
research_params.json (surfaced through config.get_research_params()),
run_cio skips the LLM call entirely and advances by floor via the
existing _fallback_selection path — closing the orphaned-flag gap where
the backtester wrote the recommendation but research never read it.
"""

from unittest.mock import patch

from agents.investment_committee import ic_cio


def _make_candidate(ticker, quant_score, qual_score, sector="Technology"):
    return {
        "ticker": ticker,
        "team_id": sector.lower(),
        "quant_score": quant_score,
        "qual_score": qual_score,
        "bull_case": "",
        "bear_case": "",
        "catalysts": [],
        "conviction": "stable",
        "quant_rationale": "",
    }


def test_run_cio_bypasses_llm_when_cio_mode_deterministic():
    """cio_mode=deterministic short-circuits run_cio before any LLM
    construction — no ChatAnthropic instantiation, no structured call."""
    candidates = [
        _make_candidate("AAA", 80, 70),
        _make_candidate("BBB", 60, 50),
        _make_candidate("CCC", 40, 30),
    ]

    with patch.object(
        ic_cio, "get_research_params", return_value={"cio_mode": "deterministic"}
    ), patch.object(ic_cio, "ChatAnthropic") as mock_llm:
        result = ic_cio.run_cio(
            candidates=candidates,
            macro_context={},
            sector_ratings={},
            current_population=[],
            open_slots=5,
            exits=[],
            run_date="2026-07-04",
            max_new_entrants=10,
            min_new_entrants=1,
        )

    mock_llm.assert_not_called()
    assert result["advanced_tickers"], "deterministic mode must still advance floor candidates"
    assert all(
        "Fallback (LLM unusable)" in d["rationale"]
        for d in result["decisions"]
        if d["decision"] == "ADVANCE"
    )


def test_run_cio_uses_llm_path_when_cio_mode_absent():
    """Default/absent cio_mode must NOT trip the bypass — the flag is a
    no-op until the backtester actually sets it."""
    candidates = [_make_candidate("AAA", 80, 70)]

    with patch.object(ic_cio, "get_research_params", return_value={"cio_mode": ""}), patch.object(
        ic_cio, "ChatAnthropic"
    ) as mock_llm:
        try:
            ic_cio.run_cio(
                candidates=candidates,
                macro_context={},
                sector_ratings={},
                current_population=[],
                open_slots=5,
                exits=[],
                run_date="2026-07-04",
                max_new_entrants=10,
                min_new_entrants=1,
            )
        except Exception:
            pass  # the mocked LLM has no real structured-output behavior; reaching it is what we assert

    mock_llm.assert_called_once()
