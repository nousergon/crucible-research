"""Producer contract test for scoring/universe_membership.py.

Pins the ``universe_membership/{date}/membership.json`` artifact that the
predictor resolves its daily scoring universe from (alpha-engine-predictor
``inference/stages/load_universe.py``) and the dashboard's churn view reads
(crucible-dashboard ``loaders/universe_churn.py``). Locks:

  1. Artifact shape + schema_version + the named predictor cut.
  2. Rank table: rank 1 = most attractive, deterministic tie-break, unrankable
     names excluded (never coerced to a score).
  3. Cut membership: scanner cut passed through verbatim + deduped; rank cuts
     sized exactly N and drawn in rank order; every cut list SORTED (set
     semantics) so a consumer diffing week-over-week sees membership changes,
     not rank churn.
  4. FAIL-LOUD on empty inputs — the defect this artifact exists to prevent is
     a silently-empty/stale membership, so an empty cut must raise, never write.
  5. Both S3 keys written (dated + latest sidecar) with identical bodies.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    PREDICTOR_UNIVERSE_CUT,
    SCHEMA_VERSION,
    UniverseMembershipError,
    attractiveness_from_board,
    build_universe_membership,
    write_universe_membership_to_s3,
)

_RUN_DATE = "2026-07-24"


def _attractiveness(n: int = 100) -> dict[str, float]:
    """``n`` synthetic names, descending score — T000 most attractive."""
    return {f"T{i:03d}": float(100 - i) for i in range(n)}


def _scanner_cut() -> list[str]:
    """A 60-name scanner cut that deliberately does NOT match the rank order —
    the gate cut and the attractiveness ranking are independent by construction
    (that divergence is the thing the artifact exists to make measurable)."""
    return [f"T{i:03d}" for i in range(40, 100)]


def _membership(**kwargs) -> dict:
    return build_universe_membership(
        _RUN_DATE, _scanner_cut(), _attractiveness(), generated_at="2026-07-24T12:00:00+00:00", **kwargs
    )


# ── 1. Envelope ──────────────────────────────────────────────────────────────


def test_envelope_shape_and_schema_version():
    m = _membership()
    assert m["schema_version"] == SCHEMA_VERSION
    assert m["run_date"] == _RUN_DATE
    assert m["producer"].endswith("scoring/universe_membership.py")
    assert m["generated_at"] == "2026-07-24T12:00:00+00:00"
    assert m["universe_count"] == 100
    # The predictor's cut is NAMED in the artifact, not hardcoded downstream —
    # changing which cut drives inference must be a producer-side edit.
    assert m["predictor_universe_cut"] == PREDICTOR_UNIVERSE_CUT
    assert m["predictor_universe_cut"] in m["cuts"]


def test_every_cut_carries_basis_size_and_provenance():
    for name, cut in _membership()["cuts"].items():
        assert cut["basis"] in ("scanner_gate", "attractiveness_rank"), name
        assert cut["size"] == len(cut["tickers"]), name
        assert cut["source"], name


# ── 2. Rank table ────────────────────────────────────────────────────────────


def test_rank_one_is_most_attractive():
    ranks = _membership()["ranks"]
    assert ranks["T000"]["attractiveness_rank"] == 1
    assert ranks["T099"]["attractiveness_rank"] == 100


def test_tie_break_is_deterministic_by_ticker():
    m = build_universe_membership(_RUN_DATE, ["AAA"], {"BBB": 50.0, "AAA": 50.0, "CCC": 50.0})
    order = [t for t, _ in sorted(m["ranks"].items(), key=lambda kv: kv[1]["attractiveness_rank"])]
    assert order == ["AAA", "BBB", "CCC"]


def test_unrankable_names_are_absent_not_zeroed():
    # A name with no attractiveness must not appear at all. Coercing it to 0.0
    # would rank it as the LEAST attractive name rather than as uncovered.
    board = {
        "stocks": [
            {"ticker": "AAA", "attractiveness_score": 90.0},
            {"ticker": "BBB", "attractiveness_score": None},
            {"ticker": "CCC"},
        ]
    }
    scores = attractiveness_from_board(board)
    assert scores == {"AAA": 90.0}
    m = build_universe_membership(_RUN_DATE, ["AAA"], scores)
    assert m["universe_count"] == 1
    assert set(m["ranks"]) == {"AAA"}


# ── 3. Cut membership ────────────────────────────────────────────────────────


def test_scanner_cut_passes_through_verbatim():
    m = _membership()
    assert m["cuts"]["scanner_candidates"]["tickers"] == sorted(_scanner_cut())
    assert m["cuts"]["scanner_candidates"]["basis"] == "scanner_gate"


def test_scanner_cut_is_deduped():
    m = build_universe_membership(_RUN_DATE, ["AAA", "BBB", "AAA"], _attractiveness())
    cut = m["cuts"]["scanner_candidates"]
    assert cut["tickers"] == ["AAA", "BBB"]
    assert cut["size"] == 2


@pytest.mark.parametrize("n", [25, 60])
def test_rank_cuts_are_exactly_n_top_ranked_names(n):
    m = _membership()
    cut = m["cuts"][f"attractiveness_top_{n}"]
    assert cut["size"] == n
    assert cut["tickers"] == sorted(f"T{i:03d}" for i in range(n))
    assert cut["basis"] == "attractiveness_rank"


def test_rank_cut_truncates_to_universe_when_smaller_than_n():
    # A 10-name universe cannot yield a 25-name cut; it must yield 10, not pad.
    m = build_universe_membership(_RUN_DATE, ["T000"], _attractiveness(10))
    assert m["cuts"]["attractiveness_top_25"]["size"] == 10
    assert m["cuts"]["attractiveness_top_60"]["size"] == 10


def test_cut_tickers_are_sorted_not_rank_ordered():
    # Set semantics: a consumer diffing consecutive weeks must see membership
    # changes, not reordering. Rank order stays recoverable from `ranks`.
    for cut in _membership()["cuts"].values():
        assert cut["tickers"] == sorted(cut["tickers"])


def test_scanner_cut_and_rank_cut_are_independent():
    # Guards against a refactor that quietly derives one cut from the other —
    # their divergence is the measurement this artifact exists to enable.
    m = _membership()
    scanner = set(m["cuts"]["scanner_candidates"]["tickers"])
    rank60 = set(m["cuts"]["attractiveness_top_60"]["tickers"])
    assert scanner != rank60


# ── 4. Fail loud ─────────────────────────────────────────────────────────────


def test_empty_scanner_cut_raises():
    with pytest.raises(UniverseMembershipError, match="scanner_tickers is empty"):
        build_universe_membership(_RUN_DATE, [], _attractiveness())


def test_empty_attractiveness_raises():
    with pytest.raises(UniverseMembershipError, match="no rankable attractiveness"):
        build_universe_membership(_RUN_DATE, _scanner_cut(), {})


# ── 5. S3 write ──────────────────────────────────────────────────────────────


class _FakeS3:
    def __init__(self):
        self.puts: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803
        self.puts[Key] = Body


def test_writes_dated_key_and_latest_sidecar_identically():
    s3 = _FakeS3()
    key = write_universe_membership_to_s3(_membership(), _RUN_DATE, bucket="test-bucket", s3_client=s3)
    assert key == f"universe_membership/{_RUN_DATE}/membership.json"
    assert set(s3.puts) == {key, "universe_membership/latest.json"}
    assert s3.puts[key] == s3.puts["universe_membership/latest.json"]
    assert json.loads(s3.puts[key])["run_date"] == _RUN_DATE


# ── 6. Incumbent challenger arm (alpha-engine-config-I4983) ──────────────────
#
# `scanner_candidates` is stored alphabetically (set semantics), so the
# incumbent's own `tech_score` ordering is NOT recoverable from it. Without
# `scanner_ranks` there is no way to ask "what would the pre-I4983 rule have
# picked at the champion's width?" — which is the comparison the champion flip
# has to be judged against.


def _tech_scores() -> dict[str, float]:
    """tech_score for the scanner cut, deliberately ANTI-correlated with
    attractiveness: T099 (least attractive) scores highest. This is not a
    contrivance — on the live 2026-07-24 cycle the scanner cut's median name
    ranked 598 of 897 on attractiveness, so the two rules genuinely disagree,
    and a fixture where they agree would not exercise the distinction."""
    cut = _scanner_cut()
    return {t: float(i) for i, t in enumerate(cut)}


def test_incumbent_arm_absent_without_tech_scores() -> None:
    """No eval log → no incumbent arm, and NO fabricated one. A backfill whose
    inputs predate the field must degrade honestly."""
    m = _membership()
    assert "scanner_top_20" not in m["cuts"]
    assert "scanner_ranks" not in m


def test_incumbent_arm_is_top_20_by_tech_score() -> None:
    m = _membership(tech_scores=_tech_scores())
    cut = m["cuts"]["scanner_top_20"]

    assert cut["basis"] == "tech_score_rank"
    assert cut["size"] == 20
    assert cut["tickers"] == sorted(cut["tickers"]), "cut lists are SORTED (set semantics)"

    # Highest tech_score wins: _tech_scores gives the LAST cut members the
    # highest scores, so the incumbent top-20 is the tail of the cut.
    assert set(cut["tickers"]) == set(_scanner_cut()[-20:])


def test_incumbent_arm_is_count_matched_to_the_champion() -> None:
    """The whole point of N=20 — an arm's win must not be confounded between
    selection rule and breadth."""
    m = _membership(tech_scores=_tech_scores())
    assert m["cuts"]["scanner_top_20"]["size"] == m["cuts"]["attractiveness_top_20"]["size"] == 20


def test_incumbent_arm_fails_loud_when_not_count_matched() -> None:
    """Runtime guard (alpha-engine-config-I4983 closes-when): a short
    tech_score table must not silently emit ``scanner_top_20`` at n<20 while
    the champion stays at 20 — that reintroduces the breadth confound the
    ruling exists to kill. Fixture-only equality asserts cannot catch a live
    partial eval log; the producer must RAISE."""
    # Only 10 scored cut members → top-by-tech would be size 10, not 20.
    partial = {t: float(i) for i, t in enumerate(_scanner_cut()[:10])}
    with pytest.raises(UniverseMembershipError, match="count-match"):
        _membership(tech_scores=partial)


def test_incumbent_arm_disagrees_with_the_champion() -> None:
    """Guard against a fixture (or a future refactor) where both arms silently
    resolve to the same names — that would make the comparison vacuous while
    every other assertion still passed."""
    m = _membership(tech_scores=_tech_scores())
    champion = set(m["cuts"]["attractiveness_top_20"]["tickers"])
    incumbent = set(m["cuts"]["scanner_top_20"]["tickers"])
    assert champion != incumbent


def test_scanner_ranks_are_scoped_to_the_cut_and_deterministic() -> None:
    m = _membership(tech_scores=_tech_scores())
    ranks = m["scanner_ranks"]

    assert set(ranks) == set(_scanner_cut()), "ranked over the CUT, not the universe"
    assert min(r["tech_score_rank"] for r in ranks.values()) == 1
    assert sorted(r["tech_score_rank"] for r in ranks.values()) == list(range(1, len(ranks) + 1))
    # Rank 1 is the highest tech_score.
    top = min(ranks.items(), key=lambda kv: kv[1]["tech_score_rank"])
    assert top[1]["tech_score"] == max(r["tech_score"] for r in ranks.values())


def test_unscored_cut_members_are_omitted_not_ranked_worst() -> None:
    """A missing tech_score must not be coerced to 0 — that would rank the name
    as the worst possible candidate rather than as absent (same contract the
    attractiveness table already holds)."""
    partial = _tech_scores()
    dropped = _scanner_cut()[:5]
    for t in dropped:
        del partial[t]

    m = _membership(tech_scores=partial)
    assert set(m["scanner_ranks"]).isdisjoint(dropped)
    assert set(m["cuts"]["scanner_top_20"]["tickers"]).isdisjoint(dropped)


def test_tech_scores_from_eval_log_projection() -> None:
    from scoring.universe_membership import tech_scores_from_eval_log

    rows = [
        {"ticker": "aapl", "tech_score": 65.89},
        {"ticker": "MSFT", "tech_score": 60.61},
        {"ticker": "NOSCORE", "tech_score": None},
        {"ticker": "BADTYPE", "tech_score": "60"},
        {"tech_score": 99.0},
    ]
    assert tech_scores_from_eval_log(rows) == {"AAPL": 65.89, "MSFT": 60.61}
    assert tech_scores_from_eval_log(None) == {}
