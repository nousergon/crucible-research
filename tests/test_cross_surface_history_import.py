"""An arm inherits its predecessor's record ACROSS surfaces
(alpha-engine-config-I11422).

`ProducerSpec.supersedes` reads `signals_shadow/{arm}/` and nothing else. That
single fact — not any property of the arms — is why `thinktank_20` carried
history on the research slot's first cycle and `attractiveness_60` did not:
Think Tank's predecessor happened to publish to the SAME prefix as its
successor. The scanner cuts had been running the whole time, with 22 and 13
scored dates, on `universe_membership/{date}/`.

MEASURED on the live board, before and after:

    attractiveness_60   n=0  -> n=24   IR None -> -0.613
    attractiveness_20   n=0  -> n=15   IR None -> -0.1966

and the arena cycle went from `unmeasurable` with ZERO usable comparisons to
`decided`, moving the pointer on 5 paired dates over 4 weeks — including the
60-vs-20 depth comparison at 13 paired dates, which is the experiment the slot
was built to run.

WHY READING THE DATED ARTIFACT IS WHAT MAKES THIS LEGITIMATE. The arm's live
code ranks from `scanner/universe/latest.json`, a pointer whose past state is
unrecoverable — so its historical picks cannot be RECONSTRUCTED. They do not
have to be: `universe_membership/{date}/membership.json` recorded what the cut
actually WAS on each date, including its `ranks.attractiveness_rank` order. So
this imports the record rather than re-deriving it, and §3.1's "a fact to be
CHECKED, never assumed from a rename" is satisfied by construction.

VERIFIED RED: with `supersedes_cut` unset on both arms, every assertion in
`TestTheImportReachesTheBoard` fails with `n_dates_scored == 0`.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

_BUCKET = "alpha-engine-research"
_AS_OF = "2026-08-04"
_TICKERS = [f"T{i:02d}" for i in range(60)]
# Two cohort dates the cut was formed on, plus a third that HOLDS the second.
_D1, _D2, _HELD = "2026-06-01", "2026-06-08", "2026-06-09"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=_BUCKET)
        yield client


def _membership(s3, date_str: str, effective: str, tickers: list[str]) -> None:
    s3.put_object(
        Bucket=_BUCKET,
        Key=f"universe_membership/{date_str}/membership.json",
        Body=json.dumps({
            "cut_effective_date": effective,
            "cuts": {
                "attractiveness_top_60": {"tickers": sorted(tickers)},
                "attractiveness_top_20": {"tickers": sorted(tickers[:20])},
            },
            # Deliberately NOT the alphabetical order the `tickers` lists carry:
            # the rank table is the only thing that can order a cut, and an
            # import that fell back to the list order would score a Spearman
            # against the alphabet.
            "ranks": {t: {"attractiveness_rank": i + 1} for i, t in enumerate(tickers)},
        }).encode(),
    )


class _Panel:
    def __init__(self):
        self.panel: dict[str, dict[str, float]] = {}

    def put(self, date_str: str, closes: dict) -> _Panel:
        self.panel.setdefault(date_str, {}).update(
            {t: float(c) for t, c in closes.items()}
        )
        return self

    def loader(self):
        return lambda bucket, entry_dates, horizon_days, symbols=None: self.panel


def _matured(*entries: str) -> _Panel:
    panel = _Panel()
    for e in entries:
        panel.put(e, {**dict.fromkeys(_TICKERS, 100.0), "SPY": 100.0})
    for d in [f"2026-07-{d:02d}" for d in range(1, 25)]:
        panel.put(d, {
            **{t: 100.0 + (60 - i) * 0.5 for i, t in enumerate(_TICKERS)},
            "SPY": 105.0,
        })
    return panel


class TestTheCutReader:
    def test_it_reads_the_dated_rank_table_not_the_alphabetical_list(self, s3):
        """The cut's own `tickers` list is alphabetical (set semantics). Order
        must come from `ranks.attractiveness_rank` or the rank-IC is noise
        wearing the shape of a result."""
        from scoring.leaderboard_producers import _cut_picks_by_date

        _membership(s3, _D1, _D1, _TICKERS)
        got = _cut_picks_by_date(s3, _BUCKET, "attractiveness_top_60", [_D1])
        assert got[_D1].ranked == _TICKERS  # rank order, not sorted()
        assert got[_D1].rank_ordered is True

    def test_a_held_cut_is_one_observation_not_one_per_prefix(self, s3):
        """alpha-engine-config-I8269. A cut held across prefixes is ONE
        decision; counting it per prefix overstates the inherited evidence by
        the hold length."""
        from scoring.leaderboard_producers import _cut_picks_by_date

        _membership(s3, _D1, _D1, _TICKERS)
        _membership(s3, _D2, _D2, _TICKERS)
        _membership(s3, _HELD, _D2, _TICKERS)  # same effective date as _D2
        got = _cut_picks_by_date(
            s3, _BUCKET, "attractiveness_top_60", [_D1, _D2, _HELD]
        )
        assert sorted(got) == [_D1, _D2]

    def test_a_date_with_no_membership_artifact_contributes_nothing(self, s3):
        from scoring.leaderboard_producers import _cut_picks_by_date

        _membership(s3, _D1, _D1, _TICKERS)
        got = _cut_picks_by_date(
            s3, _BUCKET, "attractiveness_top_60", [_D1, "2026-06-02"]
        )
        assert sorted(got) == [_D1]

    def test_the_20_wide_cut_is_read_at_its_own_width(self, s3):
        from scoring.leaderboard_producers import _cut_picks_by_date

        _membership(s3, _D1, _D1, _TICKERS)
        got = _cut_picks_by_date(s3, _BUCKET, "attractiveness_top_20", [_D1])
        assert len(got[_D1].ranked) == 20


class TestTheImportReachesTheBoard:
    def test_both_funnel_arms_score_their_inherited_dates(self, s3):
        """The whole point: an arm that has never written a shadow of its own
        still carries a scored ladder."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        _membership(s3, _D1, _D1, _TICKERS)
        _membership(s3, _D2, _D2, _TICKERS)
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=_matured(_D1, _D2).loader()
        )
        assert res["status"] == "ok", res
        rows = {r["name"]: r for r in res["leaderboard"]["specs"]}
        assert rows["attractiveness_60"]["n_dates_scored"] == 2
        assert rows["attractiveness_20"]["n_dates_scored"] == 2
        assert rows["attractiveness_60"]["top_n"] == 60
        assert rows["attractiveness_20"]["top_n"] == 20

    def test_an_arm_with_no_declared_cut_inherits_nothing(self, s3):
        """The import is opt-in per arm. `tech_score_20` declares no
        `supersedes_cut` and must not pick up a sibling's record."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        _membership(s3, _D1, _D1, _TICKERS)
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=_matured(_D1).loader()
        )
        rows = {r["name"]: r for r in res["leaderboard"]["specs"]}
        assert rows["tech_score_20"]["n_dates_scored"] == 0

    def test_the_arms_own_picks_win_a_date_collision(self, s3):
        """`setdefault`, not overwrite: an inheritance may fill a gap the live
        arm never produced; it may never replace something the arm measured."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        _membership(s3, _D1, _D1, _TICKERS)
        # The live arm ALSO wrote a shadow on _D1, with two names only.
        s3.put_object(
            Bucket=_BUCKET,
            Key=f"signals_shadow/attractiveness_60/{_D1}/signals.json",
            Body=json.dumps({"signals": {
                "T00": {"signal": "ENTER", "score": 99.0},
                "T01": {"signal": "ENTER", "score": 98.0},
            }}).encode(),
        )
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=_matured(_D1).loader()
        )
        rows = {r["name"]: r for r in res["leaderboard"]["specs"]}
        # Its own 2-name day, not the inherited 60-name cut.
        assert rows["attractiveness_60"]["n_dates_scored"] == 1
        assert rows["attractiveness_60"]["pool_provenance"] is None

    def test_an_inherited_date_joins_the_boards_cohort(self, s3):
        """An inherited date is a cohort date of this board even though no arm
        wrote a shadow under it. Without the union the realized-return join
        would miss it and the history would load and then score zero —
        present in `by_date`, absent from every horizon, silent about why."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        _membership(s3, _D1, _D1, _TICKERS)
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=_matured(_D1).loader()
        )
        row = {r["name"]: r for r in res["leaderboard"]["specs"]}["attractiveness_60"]
        assert _D1 in row["dates_scored"]


def _point_champion_at(s3, name: str) -> None:
    s3.put_object(
        Bucket=_BUCKET,
        Key="config/producer_champion.json",
        Body=json.dumps({"schema_version": 1, "champion": name}).encode(),
    )


class TestAPromotedArmKeepsItsInheritedRecord:
    """alpha-engine-config-I11931. champion-challenger-policy.md §3: a promoted
    arm keeps its history.

    MEASURED: `attractiveness_20` scored 15 dates as a challenger on the
    2026-09-25 board, all of them inherited through `supersedes_cut`. It was
    promoted that cycle, and on the 2026-10-02 board — now the champion row —
    it scored ZERO, because the champion row read its own shadow prefix only.
    Every pairwise comparison in `arena/producer/2026-10-02.json` came back
    `common_window_too_short: 0 paired date(s)` against an incumbent with no
    series, and the cycle was `unmeasurable`.

    VERIFIED RED: on the parent commit `test_the_champion_row_scores_its_inherited_dates`
    fails with `n_dates_scored == 0` on the champion row.
    """

    def test_the_champion_row_scores_its_inherited_dates(self, s3):
        from scoring.leaderboard_producers import build_producer_leaderboard

        _membership(s3, _D1, _D1, _TICKERS)
        _membership(s3, _D2, _D2, _TICKERS)
        _point_champion_at(s3, "attractiveness_20")
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=_matured(_D1, _D2).loader()
        )
        assert res["status"] == "ok", res
        rows = {r["name"]: r for r in res["leaderboard"]["specs"]}
        champ = rows["attractiveness_20"]
        assert champ["kind"] == "champion"
        assert champ["n_dates_scored"] == 2
        assert champ["top_n"] == 20
        # The per-date series the arena pairs on is present for the champion,
        # so a challenger with the same dates has a common window to share.
        assert sorted(champ["topn_alpha_vs_population_by_date"]) == [_D1, _D2]
        assert rows["attractiveness_60"]["n_dates_scored"] == 2
        # Scored once, as the champion — never a second challenger row.
        names = [r["name"] for r in res["leaderboard"]["specs"]]
        assert names.count("attractiveness_20") == 1

    def test_the_champions_own_picks_still_win_a_date_collision(self, s3):
        """Promotion does not loosen `setdefault`: the inheritance fills a gap
        the serving arm never produced and never replaces what it measured."""
        from scoring.leaderboard_producers import build_producer_leaderboard

        _membership(s3, _D1, _D1, _TICKERS)
        _point_champion_at(s3, "attractiveness_20")
        s3.put_object(
            Bucket=_BUCKET,
            Key=f"signals_shadow/attractiveness_20/{_D1}/signals.json",
            Body=json.dumps({"signals": {
                "T00": {"signal": "ENTER", "score": 99.0},
                "T01": {"signal": "ENTER", "score": 98.0},
            }}).encode(),
        )
        res = build_producer_leaderboard(
            s3, _BUCKET, _AS_OF, closes_panel_loader=_matured(_D1).loader()
        )
        champ = {r["name"]: r for r in res["leaderboard"]["specs"]}["attractiveness_20"]
        assert champ["kind"] == "champion"
        assert champ["n_dates_scored"] == 1
        assert champ["pool_provenance"] is None
