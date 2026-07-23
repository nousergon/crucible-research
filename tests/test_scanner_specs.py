"""Unit tests for the scanner champion/challenger spec registry + shadow
artifact builder (config#1221 / config#1186)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import scanner_specs  # noqa: E402
from data.scanner_specs import (  # noqa: E402
    ScannerSpec,
    _rank_momentum_sleeve,
    build_shadow_artifacts,
    challenger_specs,
)


def _eval_log():
    # A/B/D liquidity-eligible; C failed liquidity; E eligible but has no loading.
    # tech_score included for the tech_score_momentum challenger rank function.
    return [
        {"ticker": "A", "liquidity_pass": 1, "tech_score": 80.0},
        {"ticker": "B", "liquidity_pass": 1, "tech_score": 70.0},
        {"ticker": "C", "liquidity_pass": 0, "tech_score": 90.0},
        {"ticker": "D", "liquidity_pass": 1, "tech_score": 60.0},
        {"ticker": "E", "liquidity_pass": 1},
    ]


def _loadings():
    return {
        "A": {"momentum_20d_zscore": 2.0, "return_60d_zscore": 2.0},  # mean 2.0
        "B": {"momentum_20d_zscore": 1.0, "return_60d_zscore": 1.0},  # mean 1.0
        "C": {"momentum_20d_zscore": 9.0, "return_60d_zscore": 9.0},  # ineligible
        "D": {"momentum_20d_zscore": 0.5, "return_60d_zscore": 0.5},  # mean 0.5
        # E intentionally absent — eligible but unscorable, must be dropped.
    }


def test_momentum_sleeve_ranks_eligible_by_zscore():
    out = _rank_momentum_sleeve(_eval_log(), _loadings(), {"momentum_top_n": 2})
    # top-N by mean(z(momentum_20d), z(return_60d)); C excluded (not eligible),
    # E dropped (no loading); count-matched to momentum_top_n=2.
    assert out == ["A", "B"], out


def test_momentum_sleeve_no_loadings_returns_empty():
    assert _rank_momentum_sleeve(_eval_log(), None, {"momentum_top_n": 5}) == []


def test_momentum_sleeve_partial_factor_present():
    # A name with only one of the two factors is still scored on what's present.
    lo = {"A": {"momentum_20d_zscore": 3.0}, "B": {"return_60d_zscore": 1.0}}
    out = _rank_momentum_sleeve(
        [{"ticker": "A", "liquidity_pass": 1}, {"ticker": "B", "liquidity_pass": 1}],
        lo, {"momentum_top_n": 5},
    )
    assert out == ["A", "B"], out


def _live_artifact():
    return {
        "run_date": "2026-05-29",
        "generated_at": "2026-05-30T09:00:00+00:00",
        "population_tickers": ["AAPL", "GOOG"],
        "filters_applied": {"momentum_top_n": 2, "min_avg_volume": 500000},
        "stats": {"universe_size": 903},
    }


def test_build_shadow_artifacts_schema_and_isolation():
    shadows = build_shadow_artifacts(
        _live_artifact(), _eval_log(), _loadings(), {"momentum_top_n": 2}
    )
    # After the config#1186 live cutover, momentum_sleeve is the CHAMPION
    # and runs in the live candidates/ path. The shadow path now builds the
    # tech_score_momentum challenger for comparison.
    assert "tech_score_momentum" in shadows, shadows
    a = shadows["tech_score_momentum"]
    # Parallel-to-live schema so a leaderboard can read live + shadows uniformly.
    assert a["run_date"] == "2026-05-29"
    assert a["scanner_version"] == "tech_score_momentum-v1.0"
    assert a["spec"] == {
        "name": "tech_score_momentum", "kind": "challenger",
        "ranking": scanner_specs.SCANNER_SPECS["tech_score_momentum"].description,
    }
    assert a["scanner_tickers"] == ["A", "B"]
    # population carried from live; agent_input = population ∪ picks[:50].
    assert a["population_tickers"] == ["AAPL", "GOOG"]
    assert a["agent_input_set"] == ["AAPL", "GOOG", "A", "B"]
    assert a["stats"]["post_scanner"] == 2
    assert a["stats"]["eligible_universe"] == 4  # A,B,D,E pass liquidity
    assert a["stats"]["spec_scored"] == 2


def test_build_shadow_artifacts_is_failsoft_per_spec(monkeypatch):
    def _boom(eval_log, factor_loadings, params):
        raise RuntimeError("synthetic spec failure")

    monkeypatch.setitem(
        scanner_specs.SCANNER_SPECS, "broken",
        ScannerSpec(name="broken", kind="challenger", version="v1",
                    description="always raises", rank=_boom),
    )
    # The broken spec is swallowed (logged WARN); the healthy one still emits.
    shadows = build_shadow_artifacts(
        _live_artifact(), _eval_log(), _loadings(), {"momentum_top_n": 2}
    )
    assert "broken" not in shadows
    assert "tech_score_momentum" in shadows


def test_registry_has_one_champion_and_challengers():
    champions = [s for s in scanner_specs.SCANNER_SPECS.values() if s.kind == "champion"]
    # Post config#1186 live cutover: champion (momentum_sleeve) has a rank
    # function (z-score blend), challengers also have rank functions.
    assert len(champions) == 1
    assert champions[0].name == "momentum_sleeve"
    assert champions[0].rank is not None
    assert all(s.rank is not None for s in challenger_specs())
    assert all(s.kind == "challenger" for s in challenger_specs())
