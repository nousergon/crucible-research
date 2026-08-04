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
    build_shadow_status_record,
    challenger_specs,
)


def _eval_log():
    # A/B/D liquidity-eligible; C failed liquidity; E eligible but has no loading.
    return [
        {"ticker": "A", "liquidity_pass": 1},
        {"ticker": "B", "liquidity_pass": 1},
        {"ticker": "C", "liquidity_pass": 0},
        {"ticker": "D", "liquidity_pass": 1},
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
        lo,
        {"momentum_top_n": 5},
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
    shadows, errors = build_shadow_artifacts(_live_artifact(), _eval_log(), _loadings(), {"momentum_top_n": 2})
    assert errors == {}, errors
    assert "momentum_sleeve" in shadows, shadows
    a = shadows["momentum_sleeve"]
    # Parallel-to-live schema so a leaderboard can read live + shadows uniformly.
    assert a["run_date"] == "2026-05-29"
    assert a["scanner_version"] == "momentum_sleeve-v1"
    assert a["spec"] == {
        "name": "momentum_sleeve",
        "kind": "challenger",
        "ranking": scanner_specs.SCANNER_SPECS["momentum_sleeve"].description,
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
        scanner_specs.SCANNER_SPECS,
        "broken",
        ScannerSpec(name="broken", kind="challenger", version="v1", description="always raises", rank=_boom),
    )
    # The broken spec is swallowed (logged WARN); the healthy one still emits.
    shadows, errors = build_shadow_artifacts(_live_artifact(), _eval_log(), _loadings(), {"momentum_top_n": 2})
    assert "broken" not in shadows
    assert "momentum_sleeve" in shadows
    # config#6428: the failed spec is recorded as a MISS in `errors`, not
    # merely omitted from `shadows` — champion-challenger-policy.md §3.
    assert errors == {"broken": "synthetic spec failure"}, errors
    assert "momentum_sleeve" not in errors


def test_registry_has_one_champion_and_challengers():
    champions = [s for s in scanner_specs.SCANNER_SPECS.values() if s.kind == "champion"]
    assert len(champions) == 1 and champions[0].rank is None
    assert all(s.rank is not None for s in challenger_specs())


# ── config#6428: explicit per-spec, per-cycle MISS record ────────────────────
# champion-challenger-policy.md §3: "A cycle where an arm produces no output
# is recorded as a miss, not omitted. Silent absence and a genuine zero must
# never render identically." Mirrors producers/experiment_record.py's
# experiment_record.v1 vocabulary — RED-GUARD (policy §7.4): every assertion
# below fails on pre-fix code, since neither `build_shadow_status_record` nor
# the `errors` element of `build_shadow_artifacts`'s return existed before
# this change (pre-fix `build_shadow_artifacts` returned a bare dict, so even
# the 2-tuple unpack above raises ValueError pre-fix).


def test_build_shadow_status_record_success_shape():
    spec = scanner_specs.SCANNER_SPECS["momentum_sleeve"]
    record = build_shadow_status_record(
        spec,
        "2026-08-04",
        shadow_candidates_key="candidates_shadow/momentum_sleeve/2026-08-04/candidates.json",
    )
    assert record["status"] == "complete"
    assert record["spec"] == "momentum_sleeve"
    assert record["kind"] == "challenger"
    assert record["run_date"] == "2026-08-04"
    assert record["artifacts"] == [
        {
            "name": "shadow_candidates",
            "status": "emitted",
            "key": "candidates_shadow/momentum_sleeve/2026-08-04/candidates.json",
        }
    ]


def test_build_shadow_status_record_failure_shape_is_explicit_miss():
    """The MISS record this issue exists to add: a failed cycle writes an
    explicit `status: failed` + `absent` artifact record, distinguishable
    from a genuine zero AND from silent absence."""
    spec = scanner_specs.SCANNER_SPECS["momentum_sleeve"]
    record = build_shadow_status_record(
        spec,
        "2026-08-04",
        shadow_candidates_key=None,
        error="synthetic spec failure",
    )
    assert record["status"] == "failed"
    assert record["artifacts"] == [
        {
            "name": "shadow_candidates",
            "status": "absent",
            "reason": "synthetic spec failure",
        }
    ]


def test_forced_momentum_sleeve_failure_produces_explicit_miss_end_to_end(monkeypatch):
    """Fault injection (policy §7.4 red-guard): force momentum_sleeve's
    rank() to raise, drive build_shadow_artifacts's real `errors` output
    through build_shadow_status_record exactly as
    lambda/scanner_handler.py's shadow block does, and assert the resulting
    record is an explicit miss — never the pre-fix behavior of silently
    omitting the date from candidates_shadow/."""

    def _boom(eval_log, factor_loadings, params):
        raise RuntimeError("momentum_sleeve forced failure")

    original = scanner_specs.SCANNER_SPECS["momentum_sleeve"]
    monkeypatch.setitem(
        scanner_specs.SCANNER_SPECS,
        "momentum_sleeve",
        ScannerSpec(
            name=original.name,
            kind=original.kind,
            version=original.version,
            description=original.description,
            rank=_boom,
        ),
    )

    shadows, errors = build_shadow_artifacts(_live_artifact(), _eval_log(), _loadings(), {"momentum_top_n": 2})

    # (a) the artifact is genuinely absent this cycle (unchanged fail-soft
    # contract — the live champion path must never be jeopardized).
    assert "momentum_sleeve" not in shadows
    # (b) but it is NOT merely omitted — it is a recorded miss.
    assert "momentum_sleeve" in errors
    assert "momentum_sleeve forced failure" in errors["momentum_sleeve"]

    record = build_shadow_status_record(
        scanner_specs.SCANNER_SPECS["momentum_sleeve"],
        "2026-08-04",
        shadow_candidates_key=shadows.get("momentum_sleeve"),
        error=errors.get("momentum_sleeve"),
    )
    assert record["status"] == "failed"
    assert record["artifacts"][0]["status"] == "absent"
    assert "momentum_sleeve forced failure" in record["artifacts"][0]["reason"]
