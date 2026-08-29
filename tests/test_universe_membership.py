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
  5. All THREE S3 keys written with identical bodies — the immutable
     ``runs/{stamp}.json`` copy plus the two pointers — and two writes on one
     run_date keeping both run copies (alpha-engine-config-I6785).
  6. ``turnover``: per-cut retained/added/dropped against the prior WRITE, null
     rather than zeros when there is nothing to compare against.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring.universe_membership import (  # noqa: E402
    _RANK_CUTS,
    ATTRACTIVENESS_FEED_TOP_N,
    CHAMPION_CUT,
    FEED_CUT_NAME,
    FUNNEL_CONSUMER_THINKTANK,
    PREDICTOR_UNIVERSE_CUT,
    PROMOTABLE_CUTS,
    SCHEMA_VERSION,
    SLOT_ARMS,
    TECH_SCORE_RANKS_FIELD,
    UniverseMembershipError,
    assert_population_parity,
    assert_rank_tables_cover_promotable_cuts,
    attractiveness_from_board,
    build_universe_membership,
    compute_turnover,
    declared_cut_for,
    promotion_ineligibility_from_rank_tables,
    rank_table_for_cut,
    run_stamp,
    scanned_universe_from_eval_log,
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
        assert cut["basis"] in (
            "scanner_champion_rank",
            "attractiveness_rank",
            "tech_score_rank",
        ), name
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
    assert m["cuts"][CHAMPION_CUT]["tickers"] == sorted(_scanner_cut())
    # NOT "scanner_gate": the live cut is ranked by the scanner slot's champion
    # arm, not by the gate's tech_score, and has been since the 2026-07-22
    # cutover (alpha-engine-config-I7808).
    assert m["cuts"][CHAMPION_CUT]["basis"] == "scanner_champion_rank"


def test_scanner_cut_is_deduped():
    m = build_universe_membership(_RUN_DATE, ["AAA", "BBB", "AAA"], _attractiveness())
    cut = m["cuts"][CHAMPION_CUT]
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
    scanner = set(m["cuts"][CHAMPION_CUT]["tickers"])
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


def test_writes_immutable_run_copy_plus_both_pointers_identically():
    s3 = _FakeS3()
    m = _membership()
    key = write_universe_membership_to_s3(m, _RUN_DATE, bucket="test-bucket", s3_client=s3)
    assert key == f"universe_membership/{_RUN_DATE}/membership.json"
    run_key = f"universe_membership/{_RUN_DATE}/runs/{run_stamp(m['generated_at'])}.json"
    assert set(s3.puts) == {key, run_key, "universe_membership/latest.json"}
    assert s3.puts[key] == s3.puts["universe_membership/latest.json"] == s3.puts[run_key]
    assert json.loads(s3.puts[key])["run_date"] == _RUN_DATE


def test_two_writes_on_one_run_date_keep_both_run_copies():
    """The defect I6785 records: the weekday preopen Scanner and the
    postclose-chained exercise Scanner both normalise to the same trading day,
    so the pointer keys hold only the LAST cut of the day. Measured 2026-08-07:
    the surviving membership.json postdated the predictions it fed by 17h and
    was missing four names those predictions were stamped with. The immutable
    copies must survive that."""
    s3 = _FakeS3()
    morning = _membership()
    morning["generated_at"] = "2026-07-24T12:49:30+00:00"
    evening = _membership()
    evening["generated_at"] = "2026-07-25T03:13:01+00:00"
    evening["cuts"]["attractiveness_top_20"]["tickers"] = ["ZZZZ"]

    write_universe_membership_to_s3(morning, _RUN_DATE, bucket="b", s3_client=s3)
    write_universe_membership_to_s3(evening, _RUN_DATE, bucket="b", s3_client=s3)

    run_keys = sorted(k for k in s3.puts if "/runs/" in k)
    assert run_keys == [
        f"universe_membership/{_RUN_DATE}/runs/20260724T124930Z.json",
        f"universe_membership/{_RUN_DATE}/runs/20260725T031301Z.json",
    ]
    # The pointer holds the evening cut; the morning cut is still recoverable.
    pointer = json.loads(s3.puts[f"universe_membership/{_RUN_DATE}/membership.json"])
    assert pointer["cuts"]["attractiveness_top_20"]["tickers"] == ["ZZZZ"]
    recovered = json.loads(s3.puts[run_keys[0]])
    assert recovered["cuts"]["attractiveness_top_20"]["tickers"] != ["ZZZZ"]


def test_run_stamp_normalises_to_utc_so_the_prefix_sorts_chronologically():
    # An operator reads this prefix with `aws s3 ls`. Two runs stamped in
    # different offsets must not sort into the wrong order.
    assert run_stamp("2026-08-10T12:49:30+00:00") == "20260810T124930Z"
    assert run_stamp("2026-08-10T05:49:30-07:00") == "20260810T124930Z"
    assert run_stamp("2026-08-10T12:49:30") == "20260810T124930Z"


def test_run_stamp_falls_back_rather_than_dropping_the_write():
    # An odd key is recoverable; a skipped immutable write is not.
    assert run_stamp("not-a-timestamp") == "notatimestamp"
    assert run_stamp(None) == "None"


# ── 5b. Turnover (alpha-engine-config-I6785) ─────────────────────────────────


def _prior_membership(
    tickers_top20: list[str], *, run_date="2026-07-17", generated_at="2026-07-17T12:00:00+00:00"
) -> dict:
    m = _membership()
    m["run_date"] = run_date
    m["generated_at"] = generated_at
    m["cuts"]["attractiveness_top_20"]["tickers"] = sorted(tickers_top20)
    return m


def test_turnover_counts_retained_added_dropped_against_the_prior_write():
    current = _membership()
    now = current["cuts"]["attractiveness_top_20"]["tickers"]
    prior = _prior_membership(now[:15] + ["OLD1", "OLD2", "OLD3", "OLD4", "OLD5"])

    block = compute_turnover(current, prior)
    cut = block["per_cut"]["attractiveness_top_20"]
    assert block["prior_run_date"] == "2026-07-17"
    assert block["prior_generated_at"] == "2026-07-17T12:00:00+00:00"
    assert cut["size"] == len(now)
    assert cut["retained"] == 15
    assert cut["added"] == len(now) - 15
    assert cut["dropped"] == 5
    assert cut["retention_pct"] == 75.0


def test_turnover_is_null_not_zeros_when_there_is_no_prior():
    # Zeros would render as "nothing changed" and invent a 100%-retention point
    # at the head of every series. Absence of a comparison is its own state.
    assert compute_turnover(_membership(), None) is None


def test_turnover_is_null_when_the_prior_pointer_is_this_same_artifact():
    # A re-write of the same run must not publish a fabricated 100% retention.
    m = _membership()
    assert compute_turnover(m, dict(m)) is None


def test_turnover_skips_cuts_absent_from_either_side():
    # A cut appearing or disappearing between writes is a producer schema
    # change; reporting it as 100% churn would misattribute it to the market.
    current = _membership()
    prior = _prior_membership(current["cuts"]["attractiveness_top_20"]["tickers"])
    del prior["cuts"]["attractiveness_top_60"]
    block = compute_turnover(current, prior)
    assert "attractiveness_top_60" not in block["per_cut"]
    assert "attractiveness_top_20" in block["per_cut"]


def test_build_attaches_turnover_and_defaults_it_to_null():
    assert build_universe_membership(_RUN_DATE, _scanner_cut(), _attractiveness())["turnover"] is None
    prior = _prior_membership(["AAAA"])
    built = build_universe_membership(_RUN_DATE, _scanner_cut(), _attractiveness(), prior=prior)
    assert built["turnover"]["prior_run_date"] == "2026-07-17"


# ── 6. Incumbent challenger arm (alpha-engine-config-I4983) ──────────────────
#
# The scanner's champion cut (`CHAMPION_CUT`) is stored alphabetically (set
# semantics), so the incumbent's own `tech_score` ordering is NOT recoverable
# from it. Without
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


def test_champion_cut_is_nested_inside_the_feed_cut() -> None:
    """The funnel invariant (alpha-engine-config-I6630). Downstream evidence
    ingestion scopes to the feed cut, so the champion must be its head or the
    scored names are scored on cold context."""
    m = _membership(tech_scores=_tech_scores())
    champion = set(m["cuts"][PREDICTOR_UNIVERSE_CUT]["tickers"])
    feed = set(m["cuts"][FEED_CUT_NAME]["tickers"])
    assert champion <= feed
    # Not vacuous: the feed cut is genuinely wider than the champion.
    assert len(feed) > len(champion)


def test_feed_cut_name_is_derived_from_the_declared_width() -> None:
    """Guards the pair the consumer resolves against. A hand-edited cut name
    that no longer matches ``ATTRACTIVENESS_FEED_TOP_N`` would leave
    nousergon-data's ``_rag_scope.SCOPE_CUT`` resolving a cut this producer
    never writes — and the resolver raises rather than degrading, so the whole
    corpus fill stops."""
    assert FEED_CUT_NAME == f"attractiveness_top_{ATTRACTIVENESS_FEED_TOP_N}"
    assert ATTRACTIVENESS_FEED_TOP_N in _RANK_CUTS


def test_funnel_invariant_fails_loud_when_the_champion_escapes_the_feed_cut(
    monkeypatch,
) -> None:
    """Runtime guard, not a fixture equality. The live 2026-08-07 defect was a
    champion drawn from one ranking and a downstream scope cut drawn from
    another; nothing failed because nothing asserted the two were related.

    Simulated here by pointing the champion at the tech_score cut — the exact
    pre-I4983 shape — while the feed cut stays an attractiveness rank cut."""
    import scoring.universe_membership as mod

    monkeypatch.setattr(mod, "PREDICTOR_UNIVERSE_CUT", "scanner_top_20")
    with pytest.raises(UniverseMembershipError, match="funnel invariant"):
        _membership(tech_scores=_tech_scores())


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


# ── 8. Cut refresh cadence (alpha-engine-config-I6666) ───────────────────────
#
# Brian's ruling 2026-08-08: the cut stays on its current daily refresh, but
# moving it to weekly must be a one-switch change. What the switch changes is
# how often the CUT is re-derived — never whether the artifact is written. A
# carry-forward run still writes both keys, so `universe_membership_latest`
# stays honest at cadence `weekday_sf` (config-I6651) and a MISSED Scanner run
# stays distinguishable from a deliberately held cut.

from scoring.universe_membership import (  # noqa: E402
    CADENCE_DAILY,
    CADENCE_WEEKLY,
    DEFAULT_CUT_REFRESH_CADENCE,
    carry_forward_cuts,
    compute_and_write_universe_membership,
    cut_refresh_cadence,
    read_latest_membership,
    should_recut,
)


class _ReadableFakeS3(_FakeS3):
    """`_FakeS3` that can also serve what was put — the carry-forward path
    reads `latest.json` back."""

    def __init__(self, seed: dict | None = None):
        super().__init__()
        if seed is not None:
            self.puts["universe_membership/latest.json"] = json.dumps(seed).encode()

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key not in self.puts:
            raise NoSuchKey("nope")

        class _Body:
            def __init__(self, b):
                self._b = b

            def read(self):
                return self._b

        return {"Body": _Body(self.puts[Key])}


class NoSuchKey(Exception):
    """Named to match botocore's generated ``S3.Client.exceptions.NoSuchKey``,
    which is what the production path narrows on. A differently-named stand-in
    would make this test pass against code that cannot recognise the real
    error."""


def _prior(effective_date: str) -> dict:
    m = _membership()
    m["cut_effective_date"] = effective_date
    return m


def test_default_cadence_is_weekly(monkeypatch):
    """Brian's ruling 2026-08-27: the cut holds for a week.

    The DEFAULT carries the ruling, not the env var. Before this, the cut was
    weekly only because the Scanner Lambda happened to be invoked weekly
    (alpha-engine-config-I7811 took it off the weekday SF) — the cadence
    setting still said ``daily``, so a second invocation inside one ISO week
    re-formed the whole membership. The 2026-08-22 cycle ran four times.
    """
    monkeypatch.delenv("SCANNER_CUT_REFRESH_CADENCE", raising=False)
    assert DEFAULT_CUT_REFRESH_CADENCE == CADENCE_WEEKLY
    assert cut_refresh_cadence() == CADENCE_WEEKLY


def test_repeat_invocation_inside_one_week_does_not_recut(monkeypatch):
    """The property the default flip actually buys: idempotence.

    Under ``daily`` this was False on every call, so re-running the weekly SF
    (a routine recovery action — 2026-08-22 saw three re-runs after two
    failures) silently re-formed the cut each time and clobbered the prior
    membership under the same pointer keys.
    """
    monkeypatch.delenv("SCANNER_CUT_REFRESH_CADENCE", raising=False)
    prior = _prior("2026-08-21")  # Friday, ISO week 34
    # Same-day re-run, and every later weekday of that same ISO week.
    assert should_recut("2026-08-21", prior) is False
    assert should_recut("2026-08-22", prior) is False
    # First run of the NEXT ISO week re-cuts.
    assert should_recut("2026-08-24", prior) is True


def test_env_var_flips_the_cadence_without_a_code_change(monkeypatch):
    """The override survives for a temporary deviation from the default."""
    monkeypatch.setenv("SCANNER_CUT_REFRESH_CADENCE", "daily")
    assert cut_refresh_cadence() == CADENCE_DAILY
    monkeypatch.setenv("SCANNER_CUT_REFRESH_CADENCE", "  DAILY  ")
    assert cut_refresh_cadence() == CADENCE_DAILY


def test_unrecognised_cadence_raises_rather_than_defaulting(monkeypatch):
    """A typo silently selecting daily would be indistinguishable from the
    setting working."""
    monkeypatch.setenv("SCANNER_CUT_REFRESH_CADENCE", "weeky")
    with pytest.raises(UniverseMembershipError):
        cut_refresh_cadence()


def test_daily_cadence_always_recuts():
    assert should_recut("2026-08-04", _prior("2026-08-03"), CADENCE_DAILY) is True
    assert should_recut("2026-08-04", _prior("2026-08-04"), CADENCE_DAILY) is True


def test_weekly_cadence_carries_forward_within_an_iso_week():
    # 2026-08-03 Mon … 2026-08-07 Fri are one ISO week.
    assert should_recut("2026-08-04", _prior("2026-08-03"), CADENCE_WEEKLY) is False
    assert should_recut("2026-08-07", _prior("2026-08-03"), CADENCE_WEEKLY) is False


def test_weekly_cadence_recuts_on_the_first_run_of_a_new_iso_week():
    # 2026-08-08 Sat is still the same ISO week as 08-03; 08-10 Mon is the next.
    assert should_recut("2026-08-08", _prior("2026-08-03"), CADENCE_WEEKLY) is False
    assert should_recut("2026-08-10", _prior("2026-08-03"), CADENCE_WEEKLY) is True


def test_weekly_cadence_recuts_when_there_is_no_prior_cut():
    assert should_recut("2026-08-04", None, CADENCE_WEEKLY) is True
    assert should_recut("2026-08-04", {}, CADENCE_WEEKLY) is True
    # An artifact predating the field is not a usable prior either.
    assert should_recut("2026-08-04", {"cuts": {"x": {}}}, CADENCE_WEEKLY) is True


def test_carry_forward_holds_cuts_but_keeps_todays_ranks():
    """The gap between fresh ranks and a held cut is what makes staleness
    visible; carrying the ranks too would hide it."""
    prior = _prior("2026-08-03")
    fresh = _membership()
    fresh["ranks"] = {"NEWNAME": {"attractiveness_rank": 1, "attractiveness_score": 9.9}}
    held = carry_forward_cuts(fresh, prior)
    assert held["cuts"] == prior["cuts"]
    assert held["cut_effective_date"] == "2026-08-03"
    assert held["ranks"] == fresh["ranks"]


def test_carry_forward_preserves_the_prior_predictor_cut_name():
    """Changing PREDICTOR_UNIVERSE_CUT must not take effect mid-week without a
    re-cut — the artifact names the cut, so the held artifact must keep naming
    the one its tickers actually came from."""
    prior = _prior("2026-08-03")
    prior["predictor_universe_cut"] = "some_prior_champion"
    held = carry_forward_cuts(_membership(), prior)
    assert held["predictor_universe_cut"] == "some_prior_champion"


def test_carried_forward_artifact_still_trips_the_count_match_guard():
    """The held cut is what the predictor consumes all week — guarding only the
    freshly-derived path would leave the long-lived artifact unchecked."""
    prior = _prior("2026-08-03")
    prior["cuts"] = dict(prior["cuts"])
    prior["cuts"][PREDICTOR_UNIVERSE_CUT] = {
        "basis": "attractiveness_rank",
        "size": 20,
        "tickers": [f"T{i:03d}" for i in range(20)],
        "source": "x",
    }
    prior["cuts"]["scanner_top_20"] = {
        "basis": "tech_score_rank",
        "size": 10,
        "tickers": [f"T{i:03d}" for i in range(10)],
        "source": "x",
    }
    from scoring.universe_membership import assert_cut_invariants

    with pytest.raises(UniverseMembershipError, match="count-match broken"):
        assert_cut_invariants(carry_forward_cuts(_membership(), prior), _RUN_DATE)


def test_read_latest_membership_returns_none_when_absent():
    assert read_latest_membership(bucket="b", s3_client=_ReadableFakeS3()) is None


def test_read_latest_membership_propagates_a_real_failure():
    """A swallowed read would look like 'no prior cut', re-cut mid-week, and
    quietly restore the daily churn the weekly setting exists to stop."""

    class _Denied:
        def get_object(self, *, Bucket, Key):  # noqa: N803
            raise RuntimeError("AccessDenied")

    with pytest.raises(RuntimeError):
        read_latest_membership(bucket="b", s3_client=_Denied())


def test_weekly_end_to_end_holds_the_cut_and_still_writes_both_keys(monkeypatch):
    monkeypatch.setenv("SCANNER_CUT_REFRESH_CADENCE", "weekly")
    prior = _prior("2026-07-20")  # ISO week 30
    s3 = _ReadableFakeS3(seed=prior)
    monkeypatch.setattr(
        "scoring.universe_membership.attractiveness_for_run",
        lambda run_date, **kw: _attractiveness(),
    )
    # 2026-07-24 is inside ISO week 30 -> carry forward.
    compute_and_write_universe_membership(_RUN_DATE, _scanner_cut(), bucket="b", s3_client=s3)
    written = json.loads(s3.puts[f"universe_membership/{_RUN_DATE}/membership.json"])
    assert written["cut_refresh_cadence"] == CADENCE_WEEKLY
    assert written["cut_effective_date"] == "2026-07-20"
    assert written["cuts"] == prior["cuts"]
    assert written["run_date"] == _RUN_DATE  # written every run
    assert "universe_membership/latest.json" in s3.puts


def test_daily_end_to_end_recuts_and_stamps_today(monkeypatch):
    """The daily branch still works — reachable now via the env override only."""
    monkeypatch.setenv("SCANNER_CUT_REFRESH_CADENCE", "daily")
    s3 = _ReadableFakeS3(seed=_prior("2026-07-20"))
    monkeypatch.setattr(
        "scoring.universe_membership.attractiveness_for_run",
        lambda run_date, **kw: _attractiveness(),
    )
    compute_and_write_universe_membership(_RUN_DATE, _scanner_cut(), bucket="b", s3_client=s3)
    written = json.loads(s3.puts[f"universe_membership/{_RUN_DATE}/membership.json"])
    assert written["cut_refresh_cadence"] == CADENCE_DAILY
    assert written["cut_effective_date"] == _RUN_DATE


def test_default_end_to_end_holds_the_cut_within_one_iso_week(monkeypatch):
    """Live behaviour AFTER the 2026-08-27 default flip, with no env var set.

    ``_RUN_DATE`` (2026-07-24, Fri) and the seeded prior (2026-07-20, Mon) are
    the same ISO week, so this run must CARRY the prior cut rather than form a
    new one — and must still write the artifact, so a missed Scanner run stays
    visible as a missing artifact rather than as a held cut
    (alpha-engine-config-I6651).
    """
    monkeypatch.delenv("SCANNER_CUT_REFRESH_CADENCE", raising=False)
    s3 = _ReadableFakeS3(seed=_prior("2026-07-20"))
    monkeypatch.setattr(
        "scoring.universe_membership.attractiveness_for_run",
        lambda run_date, **kw: _attractiveness(),
    )
    compute_and_write_universe_membership(_RUN_DATE, _scanner_cut(), bucket="b", s3_client=s3)
    written = json.loads(s3.puts[f"universe_membership/{_RUN_DATE}/membership.json"])
    assert written["cut_refresh_cadence"] == CADENCE_WEEKLY
    # The cut is the PRIOR week's, carried — not re-formed on this run's date.
    assert written["cut_effective_date"] == "2026-07-20"
    # ...and the artifact is still written under this run's own date.
    assert written["run_date"] == _RUN_DATE
    assert f"universe_membership/{_RUN_DATE}/membership.json" in s3.puts


# ── 7. The funnel as a READ contract (alpha-engine-config-I7842) ──────────────
#
# Producer half of the Think Tank boundary contract. Think Tank resolves its
# coverage window and its intake ranking FROM this artifact — it no longer
# re-derives either from the universe board — so the fields it reads are now
# part of the producer's contract and must fail here, on the producer, rather
# than at 05:15 in a Lambda. Consumer half: tests/test_thinktank_feed_contract.py.


def test_funnel_names_thinktank_coverage_window():
    """The producer must keep DECLARING Think Tank's window.

    If this key disappears, the consumer fails loud by design — and it fails in
    production, on the run that needed it. The point of asserting it here is
    that the deletion is caught by the producer's own suite instead.
    """
    m = _membership()
    advances = m["funnel"]["advances_to"]
    assert FUNNEL_CONSUMER_THINKTANK in advances, (
        "universe_membership stopped declaring "
        f"funnel.advances_to.{FUNNEL_CONSUMER_THINKTANK} — Think Tank resolves "
        "its coverage window from this key and has no fallback"
    )
    assert advances[FUNNEL_CONSUMER_THINKTANK] == FEED_CUT_NAME


def test_the_cut_the_funnel_names_for_thinktank_is_actually_emitted():
    """A declaration pointing at a cut that is not in ``cuts`` is a dangling
    contract: the consumer resolves a name and finds nothing behind it."""
    m = _membership()
    named = m["funnel"]["advances_to"][FUNNEL_CONSUMER_THINKTANK]
    assert named in m["cuts"]
    assert m["cuts"][named]["tickers"]
    assert declared_cut_for(m, FUNNEL_CONSUMER_THINKTANK) == named


def test_declared_cut_for_raises_when_the_declaration_is_dropped():
    m = _membership()
    del m["funnel"]["advances_to"][FUNNEL_CONSUMER_THINKTANK]
    with pytest.raises(UniverseMembershipError, match=FUNNEL_CONSUMER_THINKTANK):
        declared_cut_for(m, FUNNEL_CONSUMER_THINKTANK)


def test_declared_cut_for_raises_when_the_named_cut_disappears():
    m = _membership()
    m["cuts"].pop(FEED_CUT_NAME)
    with pytest.raises(UniverseMembershipError, match="not in ``cuts``|not in"):
        declared_cut_for(m, FUNNEL_CONSUMER_THINKTANK)


def test_rank_table_covers_the_consumers_configured_ceiling():
    """The rank map must reach as far as Think Tank's widest configured rank.

    ``exit_rank`` is read from the consumer's own shipped config rather than
    restated here: a test asserting a hardcoded 200 would keep passing after the
    consumer widened its threshold, which is exactly the drift this is for.
    """
    import yaml

    cfg = yaml.safe_load(
        open(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "thinktank.sample.yaml")
        )
    )
    coverage = (cfg.get("thinktank") or cfg).get("coverage") or {}
    widest = int(coverage.get("exit_rank") or coverage.get("rank_ceiling"))
    assert widest > 0

    m = build_universe_membership(
        _RUN_DATE,
        _scanner_cut(),
        {f"T{i:04d}": float(2000 - i) for i in range(widest + 50)},
    )
    ranks, basis = rank_table_for_cut(m, FEED_CUT_NAME, minimum_coverage=widest)
    assert basis == "attractiveness_rank"
    assert len(ranks) >= widest


def test_a_rank_table_short_of_the_universe_is_refused():
    """The tech-basis asymmetry, asserted rather than described.

    ``scanner_ranks`` covers the cut, not the universe, so a champion ranked in
    that basis cannot answer "who is rank 150". Substituting the attractiveness
    table would rank by an arm that is not the champion — the defect this whole
    contract exists to stop — so the correct behaviour is a refusal naming the
    producer-side fix (alpha-engine-config-I7843).
    """
    m = _membership()
    m["ranks"] = dict(list(m["ranks"].items())[:10])
    with pytest.raises(UniverseMembershipError, match="I7843"):
        rank_table_for_cut(m, FEED_CUT_NAME, minimum_coverage=150)


def test_a_tech_basis_cut_has_no_rank_table_and_says_so():
    m = _membership(tech_scores={t: float(i) for i, t in enumerate(_scanner_cut())})
    incumbent = next(name for name, cut in m["cuts"].items() if cut["basis"] == "tech_score_rank")
    with pytest.raises(UniverseMembershipError, match="I7843"):
        rank_table_for_cut(m, incumbent, minimum_coverage=150)


def test_every_promotable_cut_that_can_hold_the_feed_is_a_named_cut():
    """A pointer may only name an arm the artifact can actually serve."""
    m = _membership()
    # Every promotable arm is a REGISTERED arm of the slot, count-matched at the
    # feed width. Asserted against SLOT_ARMS rather than against a prefix
    # allow-list: the allow-list had to be edited every time an arm was added,
    # which is the hand-maintained-literal shape alpha-engine-config-I9272
    # retired.
    for name in PROMOTABLE_CUTS:
        assert name in SLOT_ARMS, name
        assert name.endswith(f"_{ATTRACTIVENESS_FEED_TOP_N}"), name
    assert FEED_CUT_NAME in PROMOTABLE_CUTS, (
        "the cut the funnel declares for the feed consumers is not in the "
        "promotable set — the champion pointer could then never serve it"
    )
    assert m["funnel"]["advances_to"][FUNNEL_CONSUMER_THINKTANK] in PROMOTABLE_CUTS


# ── 9. Full rank tables per promotable basis (alpha-engine-config-I7843) ──────
#
# The champion pointer can hand any PROMOTABLE cut the funnel, and the consumer
# of that slot resolves its rank ceiling in the SERVING arm's basis. Before this
# section, the only tech table was ``scanner_ranks`` — 60 names ranked within
# the champion's own cut — so a ``tech_score_top_60`` champion could not answer
# "who is rank 150" and the arm was unpromotable in practice.


def _gate_eligible() -> dict[str, float]:
    """A momentum-path tech_score projection, deliberately NARROWER than the
    attractiveness universe: ``scan_path == "momentum"`` is the incumbent rule's
    own eligibility gate, and that narrowing is a declared property of the
    basis, not a defect."""
    return {f"T{i:04d}": float(1000 - i) for i in range(250)}


def test_every_promotable_cut_basis_has_a_full_universe_rank_table():
    """The producer-side closes-when of alpha-engine-config-I7843.

    Fails if a promotable arm is emitted whose basis has no full-universe table
    — which is a promotion that would fail in a consumer, on the morning it was
    made, with the cut already live.
    """
    m = _membership(gate_eligible_tech_scores=_gate_eligible())
    emitted = [c for c in PROMOTABLE_CUTS if c in m["cuts"]]
    assert emitted, "no promotable cut was emitted — this test would be vacuous"
    for cut_name in emitted:
        basis = m["cuts"][cut_name]["basis"]
        assert basis in m["rank_tables"], (
            f"promotable cut {cut_name!r} is ranked by {basis!r}, for which the "
            f"artifact emits no full-universe rank table (has: "
            f"{sorted(m['rank_tables'])}) — alpha-engine-config-I7843"
        )
        table = m[m["rank_tables"][basis]["field"]]
        assert table, f"{basis!r} declares a field that carries no names"


def test_a_promotable_arm_without_its_rank_table_is_a_red_run():
    """The guard, not the happy path: strip the PROMOTABLE arm's table and the
    producer refuses to publish the artifact.

    Scoped to the promotable arm deliberately (alpha-engine-config-I8060). The
    rationale is that the champion pointer can hand a promotable arm the funnel
    at any time, so its rank ceiling must resolve on the morning the promotion
    lands. `tech_score_top_60` became observe-only on 2026-08-21 and can no
    longer be handed anything, so it is recorded rather than raised on — see the
    test below.
    """
    m = _membership(gate_eligible_tech_scores=_gate_eligible())
    m.pop("ranks", None)
    m.pop("rank_tables")
    with pytest.raises(UniverseMembershipError, match="I7843"):
        assert_rank_tables_cover_promotable_cuts(m, _RUN_DATE)


def test_a_non_serving_arm_without_its_rank_table_is_recorded_not_raised(caplog):
    """Losing the table costs the arm its rank-IC and nothing live, so redding a
    Scanner run over it is the wrong trade — but the loss has to be VISIBLE
    (champion-challenger-policy.md §3).

    Widened from OBSERVE-ONLY to NON-SERVING by alpha-engine-config-I9272. With
    every arm promotable, the old scoping would have redded every Scanner run
    over the two arms that publish no rank table BY DESIGN. The property the
    raise held — a promotion must not fail in a consumer on the morning it is
    made (I7843) — moved to `promotion_ineligibility_from_rank_tables`, which
    refuses the PROMOTION rather than the run.
    """
    import logging

    m = _membership(gate_eligible_tech_scores=_gate_eligible())
    m.pop(TECH_SCORE_RANKS_FIELD)
    m["rank_tables"] = {
        b: e for b, e in (m.get("rank_tables") or {}).items() if "tech_score" not in b
    }
    with caplog.at_level(logging.WARNING):
        assert_rank_tables_cover_promotable_cuts(m, _RUN_DATE)  # no raise
    assert any(
        "no full-universe rank table" in r.getMessage() for r in caplog.records
    ), caplog.text
    # And the arm is refused at the DECISION with a reason, not silently servable.
    ineligible = promotion_ineligibility_from_rank_tables(m)
    assert any("rank_table_missing" in reason for reason in ineligible.values()), ineligible


def test_the_tech_table_covers_the_universe_not_the_cut():
    m = _membership(
        tech_scores={t: float(i) for i, t in enumerate(_scanner_cut())},
        gate_eligible_tech_scores=_gate_eligible(),
    )
    # Two tables, answering two different questions — both retained.
    assert len(m["scanner_ranks"]) == 60
    assert len(m[TECH_SCORE_RANKS_FIELD]) == len(_gate_eligible())
    ranks, basis = rank_table_for_cut(m, "tech_score_top_60", minimum_coverage=200)
    assert basis == "tech_score_rank"
    assert len(ranks) >= 200


def test_the_tech_table_is_ranked_by_tech_score_not_by_attractiveness():
    """Key names are part of the contract: a table of tech_score values carrying
    ``attractiveness_rank`` keys is a lie no reader can catch."""
    m = _membership(gate_eligible_tech_scores=_gate_eligible())
    entry = next(iter(m[TECH_SCORE_RANKS_FIELD].values()))
    assert set(entry) == {"tech_score_rank", "tech_score"}
    top = min(m[TECH_SCORE_RANKS_FIELD].items(), key=lambda kv: kv[1]["tech_score_rank"])
    assert top[0] == "T0000"  # the highest tech_score, not the most attractive


def test_rank_tables_declares_width_so_a_consumer_can_check_it():
    """A consumer must be able to ASK whether a basis reaches its ceiling rather
    than discover it by being refused."""
    m = _membership(gate_eligible_tech_scores={f"T{i:04d}": float(i) for i in range(5)})
    tech = m["rank_tables"]["tech_score_rank"]
    assert tech["size"] == 5
    assert tech["population"] == m["universe_count"]
    assert tech["serves_rank_ceiling"] is False
    assert tech["eligibility"]
    assert m["rank_tables"]["attractiveness_rank"]["serves_rank_ceiling"] is True


def test_the_index_is_read_from_the_artifact_not_from_a_module_constant():
    """``rank_tables`` is the declaration; a consumer that resolved the field
    from a constant in this module would keep working after the producer moved
    the table, and silently read the wrong one."""
    m = _membership(gate_eligible_tech_scores=_gate_eligible())
    m["relocated_tech_ranks"] = m.pop(TECH_SCORE_RANKS_FIELD)
    m["rank_tables"]["tech_score_rank"]["field"] = "relocated_tech_ranks"
    ranks, basis = rank_table_for_cut(m, "tech_score_top_60", minimum_coverage=200)
    assert basis == "tech_score_rank"
    assert len(ranks) == len(_gate_eligible())


def test_an_artifact_written_before_rank_tables_still_resolves():
    """Backwards compatibility: the module constant is the fallback index."""
    m = _membership()
    m.pop("rank_tables")
    ranks, basis = rank_table_for_cut(m, FEED_CUT_NAME, minimum_coverage=50)
    assert basis == "attractiveness_rank"
    assert len(ranks) == 100


# ── 10. Population parity with the universe board (alpha-engine-config-I7844) ─
#
# ``ranks`` and the universe board are two views of ONE Scanner invocation, and
# they were built from two different sources: the board iterates the eval log
# (the SCANNED universe), while the ranks came from every key in the factor
# profiles — a file that legitimately carries Metron-supplemental and
# fundamental-only rows the scanner never evaluated. Measured 2026-08-20 on the
# live artifacts: 906 ranked vs 903 board rows, with EQR ranked 98 — inside
# Think Tank's rank_ceiling of 150 — on four of six pillars and no technical
# data at all.


def test_ranks_outside_the_scanned_universe_are_a_red_run():
    m = _membership()
    m["population_reconciliation"] = {
        "scanned_universe_size": 903,
        "ranked": 906,
        "unrankable": [],
        "ranked_outside_scanned_universe": ["AVB", "EQR", "HYOAS", "XLRE"],
    }
    with pytest.raises(UniverseMembershipError, match="I7844"):
        assert_population_parity(m, _RUN_DATE)


def test_the_reconciliation_block_records_both_sides_and_the_difference():
    scanned = [*_attractiveness(), "VMRK"]  # one scanned name with no profile row
    m = _membership(scanned_universe=scanned)
    recon = m["population_reconciliation"]
    assert recon["scanned_universe_size"] == 101
    assert recon["ranked"] == 100
    assert recon["unrankable"] == ["VMRK"]
    assert recon["ranked_outside_scanned_universe"] == []
    assert recon["unrankable_reason"]


def test_a_wholesale_unrankable_population_is_a_red_run():
    """A handful of unscorable names is normal; a broken profiles read is not,
    and must not be published as a rank table over the remains."""
    scanned = [*_attractiveness(), *[f"X{i:04d}" for i in range(50)]]
    with pytest.raises(UniverseMembershipError, match="I7844"):
        _membership(scanned_universe=scanned)


def test_a_backfill_with_no_eval_log_says_so_rather_than_asserting():
    """The historical backfill has no eval log. An unreconciled artifact that
    SAYS it is unreconciled is honest; asserting against a population the run
    does not have is not."""
    m = _membership()
    recon = m["population_reconciliation"]
    assert recon["scanned_universe_size"] is None
    assert "not asserted" in recon["note"].lower()
    assert_population_parity(m, _RUN_DATE)  # does not raise


def test_the_scanned_universe_projection_is_the_population_not_a_score_filter():
    """Every other eval-log projection in this module selects rows by a score or
    a gate. This one is the population itself — a row that failed every gate and
    carries no tech_score is still a scanned name, and the board still has a row
    for it."""
    log = [
        {"ticker": "AAA", "tech_score": 10.0, "scan_path": "momentum"},
        {"ticker": "bbb", "tech_score": None, "scan_path": None},
        {"ticker": "AAA", "tech_score": 10.0},
        {"ticker": None},
    ]
    assert scanned_universe_from_eval_log(log) == ["AAA", "BBB"]
    assert scanned_universe_from_eval_log(None) == []
