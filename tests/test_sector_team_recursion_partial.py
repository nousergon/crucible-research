"""Tests for sector-team recursion-limit handling — the 2026-05-02 fix arc.

The Saturday SF Research step halted with ``GraphRecursionError: Recursion
limit of 16 reached`` for 5 of 6 sector teams. Two underlying issues:

1. **Budget off-by-2**: ``recursion_limit = max_iterations × 2`` was correct
   before the 2026-04-30 PR 2.3 ``response_format=...`` flip; the flip
   added one post-loop LLM call that consumed the same budget. Bump to
   ``× 2 + 2``.
2. **Crash on overrun**: budget exhaustion raised through the agent and
   crashed the whole SF instead of being a degraded-but-non-fatal outcome.
   Catch ``GraphRecursionError`` separately, return ``partial=True``,
   ``error=None`` so score_aggregator accepts it.

Tests cover:
- The recursion-limit constant calculation matches the formula above.
- Recursion errors return ``partial=True`` rather than ``error``.
- Other exceptions still flow through ``error`` (preserving hard-fail
  semantics for genuine bugs).
- ``run_sector_team`` bubbles ``partial`` up to the team-level dict.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from langgraph.errors import GraphRecursionError


@pytest.fixture
def fresh_modules():
    """Force-reload the analyst modules. ``test_dry_run.py``'s sentinel
    pattern can leave MagicMocks in place for cross-test runs in some
    pytest orders; reloading guarantees we test the real functions."""
    from agents.sector_teams import quant_analyst, qual_analyst, sector_team
    importlib.reload(quant_analyst)
    importlib.reload(qual_analyst)
    importlib.reload(sector_team)
    yield
    # No teardown — next test that needs them will reload again if it
    # also depends on freshness.


# ── Recursion-limit constants ─────────────────────────────────────────────────


def test_quant_recursion_limit_is_workload_derived():
    """config#1822 (quant sibling of the qual fix): the quant ReAct budget
    is DERIVED from the live workload (screened tickers × tools), not a
    fixed constant, so it can never under-budget the bounded worst case.

    Regressing this to a fixed ``QUANT_MAX_ITERATIONS * 2 + 2`` resurrects
    the 2026-07-11 failure where the industrials team (13 tickers × ~9
    tools) exceeded the tuned budget and returned 0 picks (partial) after
    40 legitimate, non-repeating research calls."""
    from agents.sector_teams.quant_analyst import (
        _quant_recursion_limit,
        _REACT_SYNTHESIS_MARGIN_ROUNDS,
    )
    from config import QUANT_MAX_ITERATIONS

    def expected(n_tickers, n_tools):
        # Superstep math: iterations = max(configured floor, workload rounds
        # + synthesis margin); recursion_limit = 2×iterations + 2 (one LLM
        # node + one tool node per round, +2 stop-turn tail).
        rounds = n_tickers * n_tools + _REACT_SYNTHESIS_MARGIN_ROUNDS
        return max(QUANT_MAX_ITERATIONS, rounds) * 2 + 2

    # Derivation matches the documented formula across workloads.
    assert _quant_recursion_limit(1, 1) == expected(1, 1)
    assert _quant_recursion_limit(13, 9) == expected(13, 9)

    # The incident workload (13 tickers × 9 tools) budgets for every tool on
    # every ticker PLUS synthesis headroom — provably ABOVE the old fixed
    # ceiling ``QUANT_MAX_ITERATIONS * 2 + 2`` that caused the 2026-07-11
    # partial, whenever the configured floor is below the real workload.
    assert (13 * 9 + _REACT_SYNTHESIS_MARGIN_ROUNDS) > QUANT_MAX_ITERATIONS
    assert _quant_recursion_limit(13, 9) > QUANT_MAX_ITERATIONS * 2 + 2

    # Monotonic in both tickers and tools (once past the floor).
    assert _quant_recursion_limit(14, 9) > _quant_recursion_limit(13, 9)
    assert _quant_recursion_limit(13, 10) > _quant_recursion_limit(13, 9)


def test_quant_and_qual_share_the_workload_budget_chokepoint():
    """Both analysts MUST derive their ReAct budget from the single shared
    ``workload_derived_recursion_limit`` helper — no per-analyst copy of the
    superstep math that could drift and re-open the config#1822 class on the
    next sibling. Locks the chokepoint: each analyst's binding equals the
    shared helper with its own configured floor."""
    from agents.sector_teams.react_budget import (
        workload_derived_recursion_limit,
        _REACT_SYNTHESIS_MARGIN_ROUNDS as SHARED_MARGIN,
    )
    from agents.sector_teams.quant_analyst import (
        _quant_recursion_limit,
        _REACT_SYNTHESIS_MARGIN_ROUNDS as QUANT_MARGIN,
    )
    from agents.sector_teams.qual_analyst import (
        _qual_recursion_limit,
        _REACT_SYNTHESIS_MARGIN_ROUNDS as QUAL_MARGIN,
    )
    from config import QUANT_MAX_ITERATIONS, QUAL_MAX_ITERATIONS

    # Both analysts re-export the SAME margin object from the chokepoint.
    assert QUANT_MARGIN is SHARED_MARGIN
    assert QUAL_MARGIN is SHARED_MARGIN

    # Each binding is exactly the shared helper with its analyst's floor.
    assert _quant_recursion_limit(13, 9) == workload_derived_recursion_limit(
        13, 9, floor_iterations=QUANT_MAX_ITERATIONS
    )
    assert _qual_recursion_limit(5, 13) == workload_derived_recursion_limit(
        5, 13, floor_iterations=QUAL_MAX_ITERATIONS
    )


def test_qual_recursion_limit_is_workload_derived():
    """config#1822: the qual ReAct budget is DERIVED from the live
    workload (picks × tools), not a fixed constant, so it can never
    under-budget the bounded worst case. A tiny workload floors at the
    configured ``QUAL_MAX_ITERATIONS``; a large one grows past it.

    Regressing this to a fixed ``QUAL_MAX_ITERATIONS * 2 + 2`` resurrects
    the 2026-07-11 failure where 5 picks × 13 tools exceeded the tuned
    budget and 4 qual teams returned 0 assessments (partial)."""
    from agents.sector_teams.qual_analyst import (
        _qual_recursion_limit,
        _REACT_SYNTHESIS_MARGIN_ROUNDS,
    )
    from config import QUAL_MAX_ITERATIONS

    def expected(n_picks, n_tools):
        # Superstep math: iterations = max(configured floor, workload
        # rounds + synthesis margin); recursion_limit = 2×iterations + 2
        # (one LLM node + one tool node per round, +2 stop-turn tail).
        rounds = n_picks * n_tools + _REACT_SYNTHESIS_MARGIN_ROUNDS
        return max(QUAL_MAX_ITERATIONS, rounds) * 2 + 2

    # Derivation matches the documented formula across workloads.
    assert _qual_recursion_limit(1, 1) == expected(1, 1)
    assert _qual_recursion_limit(5, 13) == expected(5, 13)

    # The incident workload (5 picks × 13 tools) budgets for every tool on
    # every pick PLUS synthesis headroom — provably ABOVE the old fixed
    # ceiling ``QUAL_MAX_ITERATIONS * 2 + 2`` that caused the 2026-07-11
    # partials, whenever the configured floor is below the real workload.
    assert (5 * 13 + _REACT_SYNTHESIS_MARGIN_ROUNDS) > QUAL_MAX_ITERATIONS
    assert _qual_recursion_limit(5, 13) > QUAL_MAX_ITERATIONS * 2 + 2

    # Monotonic in both picks and tools (once past the floor).
    assert _qual_recursion_limit(6, 13) > _qual_recursion_limit(5, 13)
    assert _qual_recursion_limit(5, 14) > _qual_recursion_limit(5, 13)


# ── Quant analyst graceful degradation ────────────────────────────────────────


def _quant_kwargs():
    """Common args to invoke run_quant_analyst."""
    return {
        "team_id": "technology",
        "sector_tickers": ["AAPL", "MSFT"],
        "market_regime": "neutral",
        "price_data": {},
        "technical_scores": {},
        "run_date": "2026-05-02",
        "api_key": "test-key",
    }


def test_quant_analyst_returns_partial_on_recursion_error(fresh_modules):
    """The 2026-05-02 scenario: agent.invoke raises GraphRecursionError.
    Must return ``partial=True, error=None`` so score_aggregator treats
    as degraded-not-failed."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError(
        "Recursion limit of 18 reached"
    )

    with patch.object(_qa, "create_react_agent", return_value=fake_agent):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is None, "recursion error must NOT populate error field"
    assert result["partial"] is True
    assert result["partial_reason"] == "recursion_limit_exhausted"
    assert result["ranked_picks"] == []


def test_quant_analyst_still_errors_on_other_exceptions(fresh_modules):
    """Generic exceptions (API errors, malformed JSON, etc.) must still
    flow through the ``error`` field — preserves hard-fail semantics for
    real bugs."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = RuntimeError("some other failure")

    with patch.object(_qa, "create_react_agent", return_value=fake_agent):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is not None
    assert "RuntimeError" in result["error"]
    assert result.get("partial", False) is False


# ── Silent step-budget exhaustion (config#1822) ───────────────────────────────
#
# The 2026-07-03 weekly's defensives/financials/consumer qual teams (and
# healthcare/industrials quant, via the retry path) burned 90-102 tool
# calls and produced zero output with error=None, partial=False — invisible
# to score_aggregator's ALL-AGENTS-STRICT gate. Root cause: langgraph's
# prebuilt ReAct executor swaps in a fixed "Sorry, need more steps..."
# AIMessage and returns NORMALLY (no GraphRecursionError) once its internal
# remaining_steps guard trips — one step earlier than the graph-level
# recursion_limit crash the existing tests above cover. These tests lock
# the fix: that sentinel must be detected and treated as partial=True,
# NOT silently extracted-from (which always yields zero picks) or crashed
# on (there's nothing to catch — the call returns normally).


def _quant_agent_result_with_sentinel(n_tool_call_messages: int = 40):
    """Build a fake ``agent.invoke()`` return value whose final AI message
    is the langgraph step-budget-exhaustion sentinel, preceded by a pile
    of tool-call bookkeeping messages so ``len(tool_calls)`` is non-trivial
    (mirrors the real 90+-tool-call artifacts)."""
    from langchain_core.messages import AIMessage, ToolMessage

    messages = []
    for i in range(n_tool_call_messages):
        ai = AIMessage(content="", tool_calls=[{
            "name": "get_price_performance", "args": {"tickers": ["AAPL"]},
            "id": f"call_{i}",
        }])
        messages.append(ai)
        messages.append(ToolMessage(content="{}", tool_call_id=f"call_{i}", name="get_price_performance"))
    messages.append(AIMessage(content="Sorry, need more steps to process this request."))
    return {"messages": messages}


def test_quant_analyst_returns_partial_on_step_budget_sentinel(fresh_modules):
    """The langgraph-internal 'need more steps' bailout must be treated
    like GraphRecursionError: partial=True, error=None, empty picks — NOT
    fed into the structured-output extractor (which would just silently
    yield zero picks with no signal that anything went wrong)."""
    from agents.sector_teams import quant_analyst as _qa

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _quant_agent_result_with_sentinel()

    with patch.object(_qa, "create_react_agent", return_value=fake_agent):
        result = _qa.run_quant_analyst(**_quant_kwargs())

    assert result["error"] is None, "sentinel must NOT populate error field"
    assert result["partial"] is True
    assert result["partial_reason"] == "remaining_steps_exhausted"
    assert result["ranked_picks"] == []
    # extract_tool_calls records one entry per AI tool_call PLUS one per
    # paired ToolMessage response, so 40 (ai, tool) rounds → 80 entries.
    assert result["iterations"] == 80, "tool calls actually made must still be counted"


def test_quant_retry_does_not_fire_on_step_budget_sentinel(fresh_modules):
    """partial=True must suppress the empty-picks retry — the agent
    already spent its full budget; a same-input retry won't help."""
    from agents.sector_teams import quant_analyst as _qa

    sentinel_result = {
        "team_id": "technology",
        "ranked_picks": [], "tool_calls": [{}] * 40,
        "iterations": 40, "error": None, "partial": True,
        "partial_reason": "remaining_steps_exhausted",
    }

    with patch.object(_qa, "run_quant_analyst", return_value=sentinel_result) as mock_run:
        result = _qa.run_quant_analyst_with_retry(**_quant_kwargs())

    assert mock_run.call_count == 1, "step-budget exhaustion must NOT trigger retry"
    assert result["retry_attempted"] is False
    assert result["partial"] is True


# ── Qual analyst graceful degradation ─────────────────────────────────────────


def _qual_kwargs():
    return {
        "team_id": "technology",
        "quant_top5": [{"ticker": "AAPL"}],
        "prior_theses": {},
        "market_regime": "neutral",
        "run_date": "2026-05-02",
        "api_key": "test-key",
        "price_data": {},
    }


def test_qual_analyst_returns_partial_on_recursion_error(fresh_modules):
    from agents.sector_teams import qual_analyst as _qual

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = GraphRecursionError(
        "Recursion limit of 18 reached"
    )

    with patch.object(_qual, "create_react_agent", return_value=fake_agent):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is None
    assert result["partial"] is True
    assert result["partial_reason"] == "recursion_limit_exhausted"
    assert result["assessments"] == []


def test_qual_analyst_still_errors_on_other_exceptions(fresh_modules):
    from agents.sector_teams import qual_analyst as _qual

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = ValueError("schema validation failed")

    with patch.object(_qual, "create_react_agent", return_value=fake_agent):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is not None
    assert "ValueError" in result["error"]
    assert result.get("partial", False) is False


def test_qual_analyst_returns_partial_on_step_budget_sentinel(fresh_modules):
    """config#1822 reproduction: defensives/financials/consumer on the
    2026-07-03 weekly each hit this exact sentinel after 90-102 tool
    calls and silently produced 0 assessments with error=None,
    partial=False. Must now be tagged partial=True so score_aggregator's
    ALL-AGENTS-STRICT gate can see it instead of the team vanishing."""
    from agents.sector_teams import qual_analyst as _qual

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = _quant_agent_result_with_sentinel(
        n_tool_call_messages=45,
    )

    with patch.object(_qual, "create_react_agent", return_value=fake_agent):
        result = _qual.run_qual_analyst(**_qual_kwargs())

    assert result["error"] is None, "sentinel must NOT populate error field"
    assert result["partial"] is True
    assert result["partial_reason"] == "remaining_steps_exhausted"
    assert result["assessments"] == []
    assert result["additional_candidate"] is None
    assert result["pillar_assessments"] == {}
    # extract_tool_calls records one entry per AI tool_call PLUS one per
    # paired ToolMessage response, so 45 (ai, tool) rounds → 90 entries —
    # matches the real defensives.json artifact's iterations=90.
    assert result["iterations"] == 90


# ── sector_team aggregation: partial bubbles up ───────────────────────────────


def test_sector_team_aggregates_partial_from_quant(fresh_modules):
    """If quant returned partial, the team-level result must surface it
    via ``partial=True`` so score_aggregator sees it."""
    from agents.sector_teams.sector_team import _empty_result

    quant_partial = {
        "ranked_picks": [],
        "error": None,
        "partial": True,
        "partial_reason": "recursion_limit_exhausted",
    }
    result = _empty_result("technology", quant_output=quant_partial)
    assert result["partial"] is True
    assert "quant:recursion_limit_exhausted" in result["partial_reasons"]
    assert result["error"] is None


def test_sector_team_no_partial_when_quant_clean(fresh_modules):
    """Default behavior — clean quant means partial=False at the team level."""
    from agents.sector_teams.sector_team import _empty_result

    result = _empty_result("technology", quant_output={
        "ranked_picks": [], "error": None,
    })
    assert result["partial"] is False
    assert result["partial_reasons"] == []


# ── Retry-on-empty-picks for quant ────────────────────────────────────────────
#
# The 2026-05-04 diagnosis: ``sector_quant:financials`` produced 22 tool
# calls and emitted ``ranked_picks=[]`` cleanly (no exception, no
# recursion exhaustion). The agent gave up at the structured-extraction
# step. ``run_quant_analyst_with_retry`` re-invokes once with an
# augmented prompt that requires at least 3 picks. Tests below lock the
# retry trigger contract + observability fields.


class TestQuantRetryOnEmpty:
    def test_should_retry_returns_true_for_give_up_case(self):
        """Empty picks + iterations>0 + no error + not partial = retry."""
        from agents.sector_teams.quant_analyst import _should_retry_on_empty_picks
        give_up = {
            "ranked_picks": [],
            "iterations": 22,
            "error": None,
            "partial": False,
        }
        assert _should_retry_on_empty_picks(give_up) is True

    def test_should_retry_returns_false_when_picks_present(self):
        from agents.sector_teams.quant_analyst import _should_retry_on_empty_picks
        ok = {
            "ranked_picks": [{"ticker": "AAPL"}],
            "iterations": 5,
            "error": None,
            "partial": False,
        }
        assert _should_retry_on_empty_picks(ok) is False

    def test_should_retry_returns_false_on_recursion_exhaustion(self):
        """Recursion budget already given — retry won't help."""
        from agents.sector_teams.quant_analyst import _should_retry_on_empty_picks
        recursion = {
            "ranked_picks": [],
            "iterations": 18,
            "error": None,
            "partial": True,
            "partial_reason": "recursion_limit_exhausted",
        }
        assert _should_retry_on_empty_picks(recursion) is False

    def test_should_retry_returns_false_on_exception(self):
        """Exception path — retry won't fix a missing API key etc."""
        from agents.sector_teams.quant_analyst import _should_retry_on_empty_picks
        errored = {
            "ranked_picks": [],
            "iterations": 0,
            "error": "RuntimeError: api key missing",
            "partial": False,
        }
        assert _should_retry_on_empty_picks(errored) is False

    def test_should_retry_returns_false_when_no_iterations(self):
        """Zero iterations means tools never ran — same input would
        produce same failure. Don't retry."""
        from agents.sector_teams.quant_analyst import _should_retry_on_empty_picks
        no_iter = {
            "ranked_picks": [],
            "iterations": 0,
            "error": None,
            "partial": False,
        }
        assert _should_retry_on_empty_picks(no_iter) is False

    def test_wrapper_no_op_when_first_attempt_succeeds(self, fresh_modules):
        """When the first attempt produces picks, the wrapper must NOT
        retry — verify run_quant_analyst is called exactly once."""
        from agents.sector_teams import quant_analyst as _qa

        first_result = {
            "team_id": "technology",
            "ranked_picks": [{"ticker": "AAPL", "quant_score": 70}],
            "tool_calls": [{"tool": "screen_by_volume"}],
            "iterations": 1,
            "error": None,
            "partial": False,
        }

        with patch.object(_qa, "run_quant_analyst", return_value=first_result) as mock_run:
            result = _qa.run_quant_analyst_with_retry(**_quant_kwargs())

        assert mock_run.call_count == 1, (
            "First-attempt success must NOT trigger retry. "
            f"Got call_count={mock_run.call_count}"
        )
        assert result["retry_attempted"] is False
        assert result["retry_succeeded"] is False
        assert result["retry_first_iterations"] is None
        # Original picks pass through unchanged.
        assert result["ranked_picks"] == [{"ticker": "AAPL", "quant_score": 70}]

    def test_wrapper_retries_once_on_empty_picks_with_iterations(self, fresh_modules):
        """The give-up case: empty picks + iterations>0. Wrapper must
        invoke the inner function exactly twice, the second call MUST
        carry the retry preamble, and observability fields populate."""
        from agents.sector_teams import quant_analyst as _qa

        first_result = {
            "team_id": "financials",
            "ranked_picks": [],
            "tool_calls": [{"tool": "screen_by_volume"}] * 22,
            "iterations": 22,
            "error": None,
            "partial": False,
        }
        retry_result = {
            "team_id": "financials",
            "ranked_picks": [
                {"ticker": "JPM", "quant_score": 55},
                {"ticker": "BAC", "quant_score": 52},
                {"ticker": "WFC", "quant_score": 50},
            ],
            "tool_calls": [{"tool": "screen_by_volume"}] * 8,
            "iterations": 8,
            "error": None,
            "partial": False,
        }

        with patch.object(
            _qa, "run_quant_analyst",
            side_effect=[first_result, retry_result],
        ) as mock_run:
            kwargs = _quant_kwargs()
            kwargs["team_id"] = "financials"
            result = _qa.run_quant_analyst_with_retry(**kwargs)

        # Exactly two invocations.
        assert mock_run.call_count == 2

        # Second call must carry the retry preamble; first must NOT.
        first_call_kwargs = mock_run.call_args_list[0].kwargs
        second_call_kwargs = mock_run.call_args_list[1].kwargs
        assert first_call_kwargs.get("_retry_preamble") is None
        assert second_call_kwargs.get("_retry_preamble") is not None
        assert "RETRY NOTICE" in second_call_kwargs["_retry_preamble"]
        assert "22 tool calls" in second_call_kwargs["_retry_preamble"]

        # Observability fields populated; retry succeeded.
        assert result["retry_attempted"] is True
        assert result["retry_succeeded"] is True
        assert result["retry_first_iterations"] == 22
        assert len(result["ranked_picks"]) == 3

    def test_wrapper_records_failure_when_retry_also_returns_empty(self, fresh_modules):
        """If retry ALSO produces zero picks, we surface that explicitly
        via retry_succeeded=False rather than silently returning empty.
        Downstream behavior is unchanged (qual loop bypasses) but
        ops can see the give-up persisted across both attempts."""
        from agents.sector_teams import quant_analyst as _qa

        empty_first = {
            "team_id": "financials",
            "ranked_picks": [], "tool_calls": [{}] * 22,
            "iterations": 22, "error": None, "partial": False,
        }
        empty_retry = {
            "team_id": "financials",
            "ranked_picks": [], "tool_calls": [{}] * 6,
            "iterations": 6, "error": None, "partial": False,
        }

        with patch.object(
            _qa, "run_quant_analyst",
            side_effect=[empty_first, empty_retry],
        ):
            kwargs = _quant_kwargs()
            kwargs["team_id"] = "financials"
            result = _qa.run_quant_analyst_with_retry(**kwargs)

        assert result["retry_attempted"] is True
        assert result["retry_succeeded"] is False
        assert result["retry_first_iterations"] == 22
        assert result["ranked_picks"] == []

    def test_wrapper_does_not_retry_on_recursion_exhaustion(self, fresh_modules):
        """``partial=True`` means the agent already used its full budget;
        rerunning won't help. Retry must NOT fire."""
        from agents.sector_teams import quant_analyst as _qa

        recursion_result = {
            "team_id": "technology",
            "ranked_picks": [], "tool_calls": [],
            "iterations": _qa._quant_recursion_limit(2, 9),
            "error": None, "partial": True,
            "partial_reason": "recursion_limit_exhausted",
        }

        with patch.object(_qa, "run_quant_analyst", return_value=recursion_result) as mock_run:
            result = _qa.run_quant_analyst_with_retry(**_quant_kwargs())

        assert mock_run.call_count == 1, "Recursion exhaustion must NOT trigger retry"
        assert result["retry_attempted"] is False
        assert result["partial"] is True

    def test_wrapper_does_not_retry_on_exception_path(self, fresh_modules):
        """Exception in the first attempt populates ``error`` and gets
        ``iterations=0``. Retry must NOT fire — same input, same crash."""
        from agents.sector_teams import quant_analyst as _qa

        errored_result = {
            "team_id": "technology",
            "ranked_picks": [], "tool_calls": [],
            "iterations": 0,
            "error": "RuntimeError: api key missing",
            "partial": False,
        }

        with patch.object(_qa, "run_quant_analyst", return_value=errored_result) as mock_run:
            result = _qa.run_quant_analyst_with_retry(**_quant_kwargs())

        assert mock_run.call_count == 1, "Exception path must NOT trigger retry"
        assert result["retry_attempted"] is False
        assert result["error"] is not None
