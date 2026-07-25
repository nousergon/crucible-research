"""Unit + producer-side contract tests for ``evals/judge_outcome_ic.py`` —
the judge-score → realized-outcome validation (old ROADMAP L480 re-scope).

Locks down:
- The FROZEN cross-repo ``judge_outcome_ic`` block shape (schema_version 1)
  the crucible-evaluator consumer is built against — key sets, types, and the
  status enum, in BOTH the "ok" and "insufficient" states (mirrors the
  producer-contract pattern of ``test_signals_producer_contract.py``).
- Attributability: thesis_update:{team}:{ticker} + a parseable capture date
  → (ticker, trading-day eval_date); everything else counted as
  unattributable, skip-markers excluded entirely (not unattributable).
- Capture-calendar-date → trading-day mapping (Saturday capture joins the
  Friday score_date the research cycle stamped).
- Date-clustered Spearman statistics REUSED from scoring/leaderboard_scoring
  (per-date >= 2-pair floor; clustered t needs >= 2 dates), with the
  pure-stdlib Student-t p-values pinned against scipy reference values.
- Honest small-N: below the floors the block ships status="insufficient"
  with null metrics and real counts — never fabricated values.
- Anti-Goodhart: the module exposes NO gate surface (compute is pure and
  returns only the diagnostic block).
"""

from __future__ import annotations

import pytest
from nousergon_lib.quant.horizons import DEFAULT_POLICY

from evals.judge_outcome_ic import (
    MIN_EVAL_DATES,
    SCHEMA_VERSION,
    AttributedEval,
    AttributionResult,
    _pooled_spearman_p,
    attribute_evals,
    compute_judge_outcome_ic,
    student_t_two_sided_p,
)
from evals.outcome_store import PrimaryOutcome

# ── Fixture helpers ────────────────────────────────────────────────────────


def _eval_doc(agent_id, s3_key, dims, skip=None):
    """A persisted RubricEvalArtifact dict (see graph/state_schemas.py)."""
    return {
        "schema_version": 2,
        "run_id": "2026-06-05",
        "judge_run_id": "2606061230",
        "timestamp": "2026-06-06T13:00:00Z",
        "judged_agent_id": agent_id,
        "judged_artifact_s3_key": s3_key,
        "rubric_id": "eval_rubric_thesis_update",
        "rubric_version": "1.0.0",
        "judge_model": "claude-haiku-4-5",
        "dimension_scores": [
            {"dimension": d, "score": s, "reasoning": "r"} for d, s in dims
        ],
        "overall_reasoning": "ok",
        "judge_skip_reason": skip,
    }


def _capture_key(capture_date: str, agent_id: str) -> str:
    y, m, d = capture_date.split("-")
    return f"decision_artifacts/{y}/{m}/{d}/run1/{agent_id}.json"


def _outcome(ticker, score_date, log_alpha):
    return PrimaryOutcome(
        symbol=ticker, score_date=score_date, beat_spy=1 if (log_alpha or 0) > 0 else 0,
        stock_return=0.01, spy_return=0.005, log_alpha=log_alpha,
    )


# Two Saturday-captured cohorts → Friday trading days 2026-06-05 / 2026-06-12.
# date1 judge scores are perfectly monotone with realized alpha (IC 1.0);
# date2 has one inversion (IC 0.5) so the clustered SE is nonzero and the
# t-stat is defined (two identical ICs would give se=0 → t None by design).
_D1_CAP, _D1 = "2026-06-06", "2026-06-05"
_D2_CAP, _D2 = "2026-06-13", "2026-06-12"

_SCORES = {  # (capture_date, ticker) -> overall judge score (both dims equal)
    (_D1_CAP, "AAA"): 4.5, (_D1_CAP, "BBB"): 3.0, (_D1_CAP, "CCC"): 2.0,
    (_D2_CAP, "AAA"): 4.0, (_D2_CAP, "BBB"): 3.5, (_D2_CAP, "CCC"): 1.5,
}
_ALPHAS = {
    ("AAA", _D1): 0.10, ("BBB", _D1): 0.02, ("CCC", _D1): -0.05,
    ("AAA", _D2): 0.01, ("BBB", _D2): 0.03, ("CCC", _D2): -0.02,
}


def _attributed_fixture(n_unattributable=0):
    attributed = [
        AttributedEval(
            ticker=t,
            eval_date=_D1 if cap == _D1_CAP else _D2,
            dimension_scores={"depth": s, "grounding": s},
        )
        for (cap, t), s in _SCORES.items()
    ]
    return AttributionResult(attributed, n_unattributable, 0)


def _outcomes_fixture():
    return {
        (t, d): _outcome(t, d, a) for (t, d), a in _ALPHAS.items()
    }


# ── Pure-stdlib Student-t p — pinned against scipy references ──────────────


class TestStudentTP:
    @pytest.mark.parametrize(
        "t_stat, df, expected",  # expected = 2 * scipy.stats.t.sf(t, df)
        [
            (2.0, 10, 0.07338803477074037),
            (1.5, 3, 0.23058386524482294),
            (4.2, 7, 0.004035559925219959),
            (0.7, 29, 0.4895051486144835),
            (3.0, 1, 0.20483276469913345),
        ],
    )
    def test_matches_scipy_reference(self, t_stat, df, expected):
        assert student_t_two_sided_p(t_stat, df) == pytest.approx(expected, rel=1e-9)

    def test_symmetric_and_bounded(self):
        assert student_t_two_sided_p(0.0, 5) == pytest.approx(1.0)
        assert student_t_two_sided_p(-2.0, 10) == student_t_two_sided_p(2.0, 10)
        assert 0.0 < student_t_two_sided_p(50.0, 4) < 1e-4

    def test_rejects_nonpositive_df(self):
        with pytest.raises(ValueError):
            student_t_two_sided_p(1.0, 0)

    def test_pooled_spearman_p_reference(self):
        # scipy.stats.spearmanr([1,2,3,4,5,6],[2,1,4,3,6,5]) → p=0.0415627
        assert _pooled_spearman_p(0.8285714285714287, 6) == pytest.approx(
            0.04156268221574335, rel=1e-6,
        )

    def test_pooled_spearman_p_edge_cases(self):
        assert _pooled_spearman_p(0.5, 2) is None       # df would be 0
        assert _pooled_spearman_p(1.0, 10) == 0.0       # |r|=1 → t diverges
        assert _pooled_spearman_p(-1.0, 10) == 0.0


# ── Attribution ────────────────────────────────────────────────────────────


class TestAttribution:
    def test_thesis_update_resolves_ticker_and_trading_day(self):
        doc = _eval_doc(
            "thesis_update:technology:AAPL",
            _capture_key(_D1_CAP, "thesis_update:technology:AAPL"),
            [("depth", 4), ("grounding", 5)],
        )
        res = attribute_evals([doc])
        assert res.n_unattributable == 0
        assert len(res.attributed) == 1
        ev = res.attributed[0]
        assert ev.ticker == "AAPL"
        # Saturday capture 2026-06-06 → Friday score_date 2026-06-05 (the
        # "most recent trading day" stamping rule, lambda/handler.py).
        assert ev.eval_date == _D1
        assert ev.dimension_scores == {"depth": 4.0, "grounding": 5.0}

    def test_non_ticker_agents_counted_unattributable(self):
        docs = [
            _eval_doc("ic_cio", _capture_key(_D1_CAP, "ic_cio"), [("d", 4)]),
            _eval_doc("sector_quant:technology",
                      _capture_key(_D1_CAP, "sector_quant:technology"), [("d", 3)]),
            _eval_doc("macro_economist", None, [("d", 5)]),
        ]
        res = attribute_evals(docs)
        assert res.attributed == []
        assert res.n_unattributable == 3

    def test_missing_or_malformed_s3_key_is_unattributable(self):
        docs = [
            _eval_doc("thesis_update:technology:AAPL", None, [("d", 4)]),
            _eval_doc("thesis_update:technology:MSFT",
                      "somewhere/else/MSFT.json", [("d", 4)]),
        ]
        res = attribute_evals(docs)
        assert res.attributed == []
        assert res.n_unattributable == 2

    def test_skip_markers_excluded_not_unattributable(self):
        docs = [
            _eval_doc("thesis_update:technology:AAPL",
                      _capture_key(_D1_CAP, "thesis_update:technology:AAPL"),
                      [], skip="degenerate_input"),
            _eval_doc("ic_cio", _capture_key(_D1_CAP, "ic_cio"),
                      [], skip="precluded_by_empty_upstream"),
        ]
        res = attribute_evals(docs)
        assert res.attributed == []
        assert res.n_unattributable == 0
        assert res.n_skip_markers == 2

    def test_trading_day_capture_maps_to_itself(self):
        doc = _eval_doc(
            "thesis_update:financials:JPM",
            _capture_key("2026-06-12", "thesis_update:financials:JPM"),  # Friday
            [("d", 3)],
        )
        res = attribute_evals([doc])
        assert res.attributed[0].eval_date == "2026-06-12"


# ── Core computation + FROZEN block contract ───────────────────────────────


_BLOCK_KEYS = {
    "schema_version", "status", "horizon_days", "overall", "by_dimension",
    "n_unattributable",
}
_OVERALL_KEYS = {
    "date_ic_mean", "date_ic_t", "date_ic_p", "n_eval_dates",
    "pooled_ic", "pooled_ic_p", "n",
}
_DIMENSION_KEYS = {"date_ic_mean", "date_ic_p", "n_eval_dates"}


class TestComputeOk:
    def _block(self, n_unattributable=2):
        return compute_judge_outcome_ic(
            _attributed_fixture(n_unattributable), _outcomes_fixture(),
        )

    def test_frozen_block_shape(self):
        block = self._block()
        assert set(block) == _BLOCK_KEYS
        assert block["schema_version"] == SCHEMA_VERSION == 1
        assert block["status"] == "ok"
        assert set(block["overall"]) == _OVERALL_KEYS
        for dim_block in block["by_dimension"].values():
            assert set(dim_block) == _DIMENSION_KEYS
        assert block["n_unattributable"] == 2

    def test_horizon_is_policy_parameter_not_hardcoded(self):
        block = self._block()
        assert block["horizon_days"] == DEFAULT_POLICY.primary_horizon == 21

    def test_date_clustered_overall(self):
        # date1 IC = 1.0 (monotone), date2 IC = 0.5 (one inversion) →
        # mean .75, se .25, t 3.0, p(df=1) = 0.204833 (scipy reference).
        o = self._block()["overall"]
        assert o["n_eval_dates"] == 2
        assert o["date_ic_mean"] == pytest.approx(0.75)
        assert o["date_ic_t"] == pytest.approx(3.0)
        assert o["date_ic_p"] == pytest.approx(0.20483276469913345, rel=1e-4)

    def test_pooled_overall(self):
        # scipy.stats.spearmanr over all 6 (score, alpha) pairs.
        o = self._block()["overall"]
        assert o["n"] == 6
        assert o["pooled_ic"] == pytest.approx(0.7714285714285715, rel=1e-6)
        assert o["pooled_ic_p"] == pytest.approx(0.07239650145772594, rel=1e-4)

    def test_by_dimension_mirrors_overall_when_dims_equal(self):
        # Both dims carry the same scores in the fixture → same clustered IC.
        bd = self._block()["by_dimension"]
        assert set(bd) == {"depth", "grounding"}
        for dim_block in bd.values():
            assert dim_block["date_ic_mean"] == pytest.approx(0.75)
            assert dim_block["n_eval_dates"] == 2
            assert dim_block["date_ic_p"] == pytest.approx(0.2048, rel=1e-2)

    def test_anticorrelated_dimension_detected(self):
        # "hedging" dim scores are REVERSED vs alpha on both dates → IC -1.0
        # per date... which gives se=0 → t None; use one reversed + one mixed
        # so the sign shows through a defined clustered stat.
        attributed = []
        rev = {("AAA"): 1.0, ("BBB"): 3.0, ("CCC"): 4.5}
        mix = {("AAA"): 2.0, ("BBB"): 1.0, ("CCC"): 4.0}
        for t in ("AAA", "BBB", "CCC"):
            attributed.append(AttributedEval(t, _D1, {"hedging": rev[t]}))
            attributed.append(AttributedEval(t, _D2, {"hedging": mix[t]}))
        block = compute_judge_outcome_ic(
            AttributionResult(attributed, 0, 0), _outcomes_fixture(),
        )
        assert block["by_dimension"]["hedging"]["date_ic_mean"] < 0

    def test_duplicate_evals_average_not_double_count(self):
        # A Sonnet-tier second eval of the same (ticker, date) averages in —
        # n (pooled pairs) must stay 6, not 7.
        attribution = _attributed_fixture()
        dup = AttributedEval("AAA", _D1, {"depth": 4.5, "grounding": 4.5})
        attribution = AttributionResult(
            attribution.attributed + [dup], 0, 0,
        )
        block = compute_judge_outcome_ic(attribution, _outcomes_fixture())
        assert block["overall"]["n"] == 6


class TestComputeInsufficient:
    def test_empty_history(self):
        block = compute_judge_outcome_ic(
            AttributionResult([], 5, 3), {},
        )
        assert block["status"] == "insufficient"
        assert set(block["overall"]) == _OVERALL_KEYS  # keys always present
        assert block["overall"]["n_eval_dates"] == 0
        assert block["overall"]["n"] == 0
        assert block["overall"]["date_ic_mean"] is None
        assert block["overall"]["pooled_ic"] is None
        assert block["by_dimension"] == {}
        assert block["n_unattributable"] == 5

    def test_single_eval_date_below_cluster_floor(self):
        assert MIN_EVAL_DATES == 2
        attributed = [
            AttributedEval(t, _D1, {"depth": s})
            for (cap, t), s in _SCORES.items() if cap == _D1_CAP
        ]
        block = compute_judge_outcome_ic(
            AttributionResult(attributed, 0, 0), _outcomes_fixture(),
        )
        assert block["status"] == "insufficient"
        # honest partials: the one date's mean IC is reported, t/p are null.
        assert block["overall"]["n_eval_dates"] == 1
        assert block["overall"]["date_ic_mean"] == pytest.approx(1.0)
        assert block["overall"]["date_ic_t"] is None
        assert block["overall"]["date_ic_p"] is None

    def test_unresolved_outcomes_do_not_join(self):
        # log_alpha None = forward window not closed → pair excluded.
        outcomes = {
            (t, d): _outcome(t, d, None) for (t, d) in _ALPHAS
        }
        block = compute_judge_outcome_ic(_attributed_fixture(), outcomes)
        assert block["status"] == "insufficient"
        assert block["overall"]["n"] == 0

    def test_one_pair_per_date_below_spearman_floor(self):
        attributed = [
            AttributedEval("AAA", _D1, {"depth": 4.0}),
            AttributedEval("AAA", _D2, {"depth": 3.0}),
        ]
        block = compute_judge_outcome_ic(
            AttributionResult(attributed, 0, 0), _outcomes_fixture(),
        )
        assert block["status"] == "insufficient"
        assert block["overall"]["n_eval_dates"] == 0
